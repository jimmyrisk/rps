#!/usr/bin/env python3
"""
Sync only aliased models (Production, B, shadow1, shadow2) from MLflow to MinIO.
This keeps MinIO lean with only the 12 currently active models.
Old model versions are automatically removed from MinIO when aliases change.

Usage:
    python scripts/sync_promoted_models_to_minio.py [--clean] [--force]
    
    --clean: Remove old models from MinIO before syncing
    --force: Re-upload models even if they already exist in MinIO
"""
import sys
import argparse
from pathlib import Path
import mlflow

# Ensure we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import set_mlflow_tracking_uri_if_needed
from app.minio_sync import (
    MINIO_BUCKET,
    MODEL_CONFIGS,
    ALIASES,
    get_current_aliased_runs,
    get_all_aliased_run_ids,
    list_minio_run_ids,
    delete_run_from_minio,
    upload_model_to_minio,
    get_minio_client_with_fallback,
)


def main():
    parser = argparse.ArgumentParser(description='Sync promoted models to MinIO')
    parser.add_argument('--clean', action='store_true', 
                       help='Remove old models from MinIO before syncing')
    parser.add_argument('--force', action='store_true',
                       help='Re-upload models even if already in MinIO')
    args = parser.parse_args()
    
    # Try to import boto3 for user-friendly error if missing
    try:
        import boto3  # noqa: F401
        from botocore.exceptions import ClientError
    except ImportError:
        print("❌ boto3 not installed. Install with: pip install boto3")
        return 1

    # Setup MLflow
    set_mlflow_tracking_uri_if_needed()
    client = mlflow.tracking.MlflowClient()
    port_forward = None
    try:
        s3_client, endpoint, port_forward = get_minio_client_with_fallback()
        print(f"✅ Connected to MinIO: {endpoint}/{MINIO_BUCKET}\n")

        # Get current aliased models
        print("🔍 Resolving current model aliases from MLflow...")
        aliased_runs = get_current_aliased_runs(client)
        current_run_ids = get_all_aliased_run_ids(aliased_runs)
        print(f"   Found {len(current_run_ids)} unique runs across all aliases\n")

        # Clean old models if requested
        if args.clean:
            print("🧹 Cleaning old models from MinIO...")
            minio_run_ids = list_minio_run_ids(s3_client)
            old_run_ids = minio_run_ids - current_run_ids

            if old_run_ids:
                print(f"   Found {len(old_run_ids)} old models to remove")
                for run_id in old_run_ids:
                    deleted = delete_run_from_minio(s3_client, run_id)
                    print(f"   🗑️  Deleted run {run_id[:8]}... ({deleted} objects)")
            else:
                print(f"   ✅ No old models to clean (MinIO has {len(minio_run_ids)} models)")
            print()

        # Sync current aliased models
        print(f"🚀 Syncing {len(MODEL_CONFIGS) * len(ALIASES)} aliased models to MinIO\n")

        synced = 0
        skipped = 0
        errors = 0

        for model_name, model_type in MODEL_CONFIGS:
            print(f"📦 Processing {model_type}:")

            if model_name not in aliased_runs:
                print(f"  ⚠️  No aliases configured for {model_name}")
                continue

            for alias in ALIASES:
                if alias not in aliased_runs[model_name]:
                    print(f"  ⏭️  {alias:12s}: No alias set")
                    skipped += 1
                    continue

                run_id, version = aliased_runs[model_name][alias]

                try:
                    # Check if model already exists in MinIO
                    model_prefix = f"{run_id}/model/"
                    exists_in_minio = False

                    if not args.force:
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

                    if exists_in_minio and not args.force:
                        print(f"  ✅ {alias:12s} (v{version}): Already in MinIO (run: {run_id[:12]}...)")
                        skipped += 1
                        continue

                    # Upload to MinIO
                    file_count = upload_model_to_minio(s3_client, run_id, model_name, alias, version)
                    print(f"  ✅ {alias:12s} (v{version}): Synced {file_count} files (run: {run_id[:12]}...)")
                    synced += 1

                except Exception as e:
                    print(f"  ❌ {alias:12s}: Error - {e}")
                    errors += 1

            print()

        # Summary
        print("=" * 60)
        print(f"✅ Synced: {synced}")
        print(f"⏭️  Skipped (already in MinIO): {skipped}")
        print(f"❌ Errors: {errors}")
        print(f"📊 Total models in MinIO: {len(current_run_ids)} (12 expected)")
        print("=" * 60)

        if errors > 0:
            return 1

        if len(current_run_ids) != 12:
            print("\n⚠️  WARNING: Expected 12 models (3 types × 4 aliases)")
            print(f"   Found {len(current_run_ids)} unique runs - some aliases may not be set")

        return 0

    except Exception as exc:
        print(f"❌ Failed to sync models: {exc}")
        return 1
    finally:
        if port_forward:
            port_forward.close()


if __name__ == "__main__":
    sys.exit(main())
