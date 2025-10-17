"""
MinIO model sync utilities.
Syncs promoted models (Production/B/shadow1/shadow2) from MLflow to MinIO.
"""
import os
import logging
import tempfile
import subprocess
import shutil
import socket
import time
import atexit
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List

logger = logging.getLogger(__name__)

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.mlops-poc.svc.cluster.local:9000")
MINIO_ENDPOINT_FALLBACKS = os.getenv("MINIO_ENDPOINT_FALLBACKS", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "mlflow-artifacts")
MINIO_NAMESPACE = os.getenv("MINIO_NAMESPACE", "mlops-poc")
MINIO_SERVICE_NAME = os.getenv("MINIO_SERVICE_NAME", "minio")
MINIO_SERVICE_PORT = int(os.getenv("MINIO_SERVICE_PORT", "9000"))
MINIO_AUTO_PORT_FORWARD = os.getenv("MINIO_AUTO_PORT_FORWARD", "true").lower() == "true"
MINIO_PORT_FORWARD_LOCAL_PORT = os.getenv("MINIO_PORT_FORWARD_LOCAL_PORT", "9000")
MINIO_PORT_FORWARD_CONTEXT = os.getenv("MINIO_PORT_FORWARD_CONTEXT")

# Model configurations
MODEL_CONFIGS = [
    ('rps_bot_xgboost', 'xgboost'),
    ('rps_bot_feedforward', 'feedforward_nn'),
    ('rps_bot_mnlogit', 'multinomial_logistic'),
]

ALIASES = ['Production', 'B', 'shadow1', 'shadow2']


class _PortForwardHandle:
    """Track kubectl port-forward process and ensure cleanup."""

    def __init__(self, process: subprocess.Popen, endpoint: str, shared: bool = False):
        self.process = process
        self.endpoint = endpoint
        self.shared = shared
        atexit.register(self.close)

    def close(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            finally:
                self.process = None


_SHARED_PORT_FORWARD: Optional[_PortForwardHandle] = None


def _parse_endpoint_list() -> List[str]:
    """Return ordered list of endpoints to try for MinIO."""

    raw_primary = MINIO_ENDPOINT
    raw_fallbacks = MINIO_ENDPOINT_FALLBACKS

    endpoints: List[str] = []

    def _add(raw: str):
        for item in raw.replace(";", ",").split(","):
            value = item.strip()
            if value and value not in endpoints:
                endpoints.append(value)

    _add(raw_primary)
    _add(raw_fallbacks)

    return endpoints


def _is_port_open(host: str, port: int, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.2)
    return False


def _start_port_forward_if_possible() -> Optional[_PortForwardHandle]:
    """Attempt to start kubectl port-forward to MinIO service."""

    if not MINIO_AUTO_PORT_FORWARD:
        return None

    if shutil.which("kubectl") is None:
        logger.debug("kubectl not available - cannot port-forward MinIO")
        return None

    try:
        preferred_port = int(MINIO_PORT_FORWARD_LOCAL_PORT)
    except ValueError:
        preferred_port = MINIO_SERVICE_PORT

    local_port = preferred_port
    if not _is_port_open("127.0.0.1", local_port, timeout=0.5):
        # quick check succeeded (port closed) -> we can use preferred
        pass
    else:
        # Find a free port dynamically
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            local_port = sock.getsockname()[1]

    cmd = ["kubectl"]
    if MINIO_PORT_FORWARD_CONTEXT:
        cmd.extend(["--context", MINIO_PORT_FORWARD_CONTEXT])
    cmd.extend([
        "-n",
        MINIO_NAMESPACE,
        "port-forward",
        f"svc/{MINIO_SERVICE_NAME}",
        f"{local_port}:{MINIO_SERVICE_PORT}",
    ])

    logger.info(
        "Attempting kubectl port-forward to MinIO service %s/%s on localhost:%s",
        MINIO_NAMESPACE,
        MINIO_SERVICE_NAME,
        local_port,
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait briefly for port-forward to establish
    if not _is_port_open("127.0.0.1", local_port, timeout=8.0):
        logger.error("kubectl port-forward did not become ready within timeout")
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        return None

    endpoint = f"http://127.0.0.1:{local_port}"
    logger.info("kubectl port-forward established for MinIO at %s", endpoint)
    return _PortForwardHandle(process, endpoint)


def _create_minio_client(endpoint: str):
    import boto3

    client = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name='us-east-1'
    )
    client.head_bucket(Bucket=MINIO_BUCKET)
    return client


def _close_port_forward(handle: Optional[_PortForwardHandle]):
    if handle and not handle.shared:
        handle.close()


def get_minio_client_with_fallback():
    """Return (s3_client, endpoint, port_forward_handle)."""

    try:
        from botocore.exceptions import EndpointConnectionError
    except ImportError:
        EndpointConnectionError = Exception  # type: ignore

    global _SHARED_PORT_FORWARD

    if _SHARED_PORT_FORWARD and _SHARED_PORT_FORWARD.process and _SHARED_PORT_FORWARD.process.poll() is None:
        try:
            client = _create_minio_client(_SHARED_PORT_FORWARD.endpoint)
            logger.info("Reusing existing MinIO port-forward at %s", _SHARED_PORT_FORWARD.endpoint)
            return client, _SHARED_PORT_FORWARD.endpoint, _SHARED_PORT_FORWARD
        except Exception as exc:  # noqa: PERF203
            logger.warning("Existing port-forward unusable, restarting: %s", exc)
            _SHARED_PORT_FORWARD.close()
            _SHARED_PORT_FORWARD = None

    last_error: Optional[Exception] = None
    for endpoint in _parse_endpoint_list():
        try:
            client = _create_minio_client(endpoint)
            logger.info("Connected to MinIO endpoint %s", endpoint)
            return client, endpoint, None
        except EndpointConnectionError as exc:
            logger.warning("MinIO endpoint %s unreachable: %s", endpoint, exc)
            last_error = exc
        except Exception as exc:  # noqa: PERF203
            logger.warning("Failed to connect to MinIO endpoint %s: %s", endpoint, exc)
            last_error = exc

    # Try kubectl port-forward if direct endpoints failed
    port_forward = _start_port_forward_if_possible()
    if port_forward:
        try:
            client = _create_minio_client(port_forward.endpoint)
            port_forward.shared = True
            _SHARED_PORT_FORWARD = port_forward
            logger.info("Connected to MinIO via port-forward at %s", port_forward.endpoint)
            return client, port_forward.endpoint, port_forward
        except Exception as exc:  # noqa: PERF203
            logger.error("MinIO port-forward connection failed: %s", exc)
            port_forward.close()
            last_error = exc

    if last_error:
        raise last_error
    raise RuntimeError("Unable to connect to MinIO")


def get_current_aliased_runs(client) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """
    Get all current run_ids that have aliases assigned.
    
    Returns:
        Dict mapping model_name -> {alias: (run_id, version)}
    """
    aliased_runs = {}
    
    for model_name, model_type in MODEL_CONFIGS:
        aliased_runs[model_name] = {}
        
        for alias in ALIASES:
            try:
                mv = client.get_model_version_by_alias(model_name, alias)
                aliased_runs[model_name][alias] = (mv.run_id, mv.version)
            except Exception:
                # Alias not set - skip
                pass
    
    return aliased_runs


def get_all_aliased_run_ids(aliased_runs: Dict[str, Dict[str, Tuple[str, str]]]) -> Set[str]:
    """Extract all unique run_ids from aliased runs"""
    run_ids = set()
    for model_aliases in aliased_runs.values():
        for run_id, _ in model_aliases.values():
            run_ids.add(run_id)
    return run_ids


def list_minio_run_ids(s3_client) -> Set[str]:
    """List all run_ids currently stored in MinIO"""
    run_ids = set()
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=MINIO_BUCKET, Delimiter='/'):
            for prefix in page.get('CommonPrefixes', []):
                # Extract run_id from prefix (format: "{run_id}/")
                run_id = prefix['Prefix'].rstrip('/')
                if run_id and len(run_id) == 32:  # MLflow run_ids are 32 chars
                    run_ids.add(run_id)
    except Exception as e:
        logger.warning(f"Failed to list MinIO contents: {e}")
    
    return run_ids


def delete_run_from_minio(s3_client, run_id: str) -> int:
    """Delete all objects for a specific run_id from MinIO. Returns count of deleted objects."""
    deleted = 0
    prefix = f"{run_id}/"
    
    try:
        # List all objects with this prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=MINIO_BUCKET, Prefix=prefix):
            objects = page.get('Contents', [])
            # Delete objects one by one (MinIO doesn't support batch delete without Content-MD5)
            for obj in objects:
                try:
                    s3_client.delete_object(Bucket=MINIO_BUCKET, Key=obj['Key'])
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {obj['Key']}: {e}")
    except Exception as e:
        logger.warning(f"Failed to delete run {run_id[:8]}: {e}")
    
    return deleted


def upload_model_to_minio(s3_client, run_id: str, model_name: str, alias: str, version: str) -> int:
    """Download model from DagsHub and upload to MinIO. Returns count of uploaded files."""
    import mlflow
    
    logger.info(f"  Downloading {alias} (v{version}) from DagsHub...")
    
    file_count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = Path(tmpdir) / "model"
        
        # Download model artifacts from MLflow
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/model",
            dst_path=str(local_path.parent)
        )
        
        # Upload to MinIO
        logger.info(f"  Uploading {alias} (v{version}) to MinIO...")
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path.parent)
                s3_key = f"{run_id}/{relative_path}"
                s3_client.upload_file(
                    str(file_path),
                    MINIO_BUCKET,
                    s3_key
                )
                file_count += 1
        
        # Create alias marker file for fast lookup
        # Format: {run_id}/.alias_{alias} contains version info
        marker_key = f"{run_id}/.alias_{alias}"
        marker_content = f"{model_name}@{alias} v{version}"
        
        marker_file = Path(tmpdir) / ".alias_marker"
        marker_file.write_text(marker_content)
        s3_client.upload_file(
            str(marker_file),
            MINIO_BUCKET,
            marker_key
        )
        file_count += 1
        logger.debug(f"  Created alias marker: {marker_key}")
    
    return file_count


def sync_aliased_model_to_minio(model_name: str, alias: str, run_id: str, version: str, force: bool = False) -> bool:
    """
    Sync a single aliased model to MinIO.
    
    Args:
        model_name: MLflow model name (e.g., 'rps_bot_feedforward')
        alias: Model alias (Production, B, shadow1, shadow2)
        run_id: MLflow run ID
        version: Model version number
        force: Re-upload even if already in MinIO
        
    Returns:
        True if synced successfully, False otherwise
    """
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        logger.error("boto3 not available - cannot sync to MinIO")
        return False
    
    port_forward = None
    try:
        s3_client, endpoint, port_forward = get_minio_client_with_fallback()
    except Exception as e:  # noqa: PERF203
        logger.error(f"Failed to connect to MinIO: {e}")
        return False
    
    try:
        # Check if model already exists in MinIO
        model_prefix = f"{run_id}/model/"
        exists_in_minio = False
        
        if not force:
            try:
                response = s3_client.list_objects_v2(
                    Bucket=MINIO_BUCKET,
                    Prefix=model_prefix,
                    MaxKeys=1
                )
                if 'Contents' in response:
                    exists_in_minio = True
            except ClientError:
                pass
        
        if exists_in_minio and not force:
            logger.info(f"Model {model_name}@{alias} (v{version}) already in MinIO")
            return True
        
        try:
            file_count = upload_model_to_minio(s3_client, run_id, model_name, alias, version)
            logger.info(f"Synced {model_name}@{alias} (v{version}): {file_count} files uploaded")
            return True
        finally:
            _close_port_forward(port_forward)

    except Exception as e:
        logger.error(f"Error syncing {model_name}@{alias}: {e}")
        _close_port_forward(port_forward)
        return False


def sync_single_model_to_minio(model_type: str, alias: str, run_id: str, version: str) -> bool:
    """
    Sync a single model to MinIO without full sync overhead.
    
    Args:
        model_type: Model type (xgboost, feedforward_nn, multinomial_logistic)
        alias: Model alias (Production, B, shadow1, shadow2)
        run_id: MLflow run ID
        version: Model version number
        
    Returns:
        True if sync succeeded, False otherwise
    """
    # Map model_type to model_name
    type_to_name = {
        'xgboost': 'rps_bot_xgboost',
        'feedforward_nn': 'rps_bot_feedforward',
        'multinomial_logistic': 'rps_bot_mnlogit',
    }
    
    model_name = type_to_name.get(model_type)
    if not model_name:
        logger.error(f"Unknown model type: {model_type}")
        return False
    
    port_forward = None
    try:
        s3_client, endpoint, port_forward = get_minio_client_with_fallback()
    except Exception as e:  # noqa: PERF203
        logger.error(f"❌ Failed to establish MinIO client: {e}")
        return False

    try:
        file_count = upload_model_to_minio(s3_client, run_id, model_name, alias, version)
        logger.info(f"✅ Synced {model_type}@{alias} v{version} to MinIO ({file_count} files)")
        return True
    except Exception as e:  # noqa: PERF203
        logger.error(f"❌ Failed to sync {model_type}@{alias}: {e}")
        return False
    finally:
        _close_port_forward(port_forward)


def sync_promoted_models_to_minio(clean: bool = True, force: bool = False) -> Tuple[int, int, int]:
    """
    Sync all promoted models (Production/B/shadow1/shadow2) to MinIO.
    
    Args:
        clean: Remove old models from MinIO before syncing
        force: Re-upload models even if already in MinIO
        
    Returns:
        Tuple of (synced_count, skipped_count, error_count)
    """
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        logger.error("boto3 not available - cannot sync to MinIO")
        return (0, 0, 1)
    
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        logger.error("mlflow not available")
        return (0, 0, 1)
    
    # Setup MLflow
    from app.config import set_mlflow_tracking_uri_if_needed
    set_mlflow_tracking_uri_if_needed()
    client = MlflowClient()
    
    # Setup MinIO S3 client
    port_forward = None
    try:
        s3_client, endpoint, port_forward = get_minio_client_with_fallback()
        logger.info(f"Connected to MinIO: {endpoint}/{MINIO_BUCKET}")
    except Exception as e:  # noqa: PERF203
        logger.error(f"Failed to connect to MinIO: {e}")
        return (0, 0, 1)
    
    # Get current aliased models
    logger.info("Resolving current model aliases from MLflow...")
    aliased_runs = get_current_aliased_runs(client)
    current_run_ids = get_all_aliased_run_ids(aliased_runs)
    logger.info(f"Found {len(current_run_ids)} unique runs across all aliases")
    
    # Clean old models if requested
    if clean:
        logger.info("Cleaning old models from MinIO...")
        minio_run_ids = list_minio_run_ids(s3_client)
        old_run_ids = minio_run_ids - current_run_ids
        
        try:
            if old_run_ids:
                logger.info(f"Found {len(old_run_ids)} old models to remove")
                for run_id in old_run_ids:
                    deleted = delete_run_from_minio(s3_client, run_id)
                    logger.info(f"Deleted run {run_id[:8]}... ({deleted} objects)")
            else:
                logger.info(f"No old models to clean (MinIO has {len(minio_run_ids)} models)")
        except Exception as exc:  # noqa: PERF203
            logger.error(f"Failed during MinIO cleanup: {exc}")
            _close_port_forward(port_forward)
            return (0, 0, 1)
    
    # Sync current aliased models
    logger.info(f"Syncing {len(MODEL_CONFIGS) * len(ALIASES)} aliased models to MinIO")
    
    synced = 0
    skipped = 0
    errors = 0
    
    for model_name, model_type in MODEL_CONFIGS:
        logger.info(f"Processing {model_type}...")
        
        if model_name not in aliased_runs:
            logger.warning(f"No aliases configured for {model_name}")
            continue
        
        for alias in ALIASES:
            if alias not in aliased_runs[model_name]:
                logger.debug(f"{alias}: No alias set")
                skipped += 1
                continue
            
            run_id, version = aliased_runs[model_name][alias]
            
            try:
                # Check if model already exists in MinIO
                model_prefix = f"{run_id}/model/"
                exists_in_minio = False
                
                if not force:
                    try:
                        response = s3_client.list_objects_v2(
                            Bucket=MINIO_BUCKET,
                            Prefix=model_prefix,
                            MaxKeys=1
                        )
                        if 'Contents' in response:
                            exists_in_minio = True
                    except ClientError:
                        pass
                
                if exists_in_minio and not force:
                    logger.info(f"{alias} (v{version}): Already in MinIO (run: {run_id[:12]}...)")
                    skipped += 1
                    continue
                
                file_count = upload_model_to_minio(s3_client, run_id, model_name, alias, version)
                logger.info(f"{alias} (v{version}): Synced {file_count} files (run: {run_id[:12]}...)")
                synced += 1
                
            except Exception as e:
                logger.error(f"{alias}: Error - {e}")
                errors += 1
    
    logger.info(f"Sync complete: {synced} synced, {skipped} skipped, {errors} errors")
    logger.info(f"Total models in MinIO: {len(current_run_ids)} (12 expected)")
    
    if len(current_run_ids) != 12:
        logger.warning(f"Expected 12 models (3 types × 4 aliases), found {len(current_run_ids)}")
    
    _close_port_forward(port_forward)

    return (synced, skipped, errors)
