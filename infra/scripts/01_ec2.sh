#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/00_env.sh"

# AMI (Ubuntu 24.04)
ARCH="$(uname -m)"; [[ "$ARCH" =~ (x86_64|amd64) ]] && AP=amd64 || AP=arm64
AMI_PARAM="/aws/service/canonical/ubuntu/server/24.04/stable/current/${AP}/hvm/ebs-gp3/ami-id"
AMI_ID=$(aws --region "$REGION" ssm get-parameter --name "$AMI_PARAM" --query 'Parameter.Value' --output text)

# Security group (idempotent)
VPC_ID=$(aws --region "$REGION" ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws --region "$REGION" ec2 describe-security-groups --filters Name=group-name,Values=rps-sg --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG_ID" = "None" ]; then
  SG_ID=$(aws --region "$REGION" ec2 create-security-group --group-name rps-sg --description "RPS k3s" --vpc-id "$VPC_ID" --query 'GroupId' --output text)
  aws --region "$REGION" ec2 authorize-security-group-ingress --group-id "$SG_ID" --ip-permissions '[
    {"IpProtocol":"tcp","FromPort":22,"ToPort":22,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]},
    {"IpProtocol":"tcp","FromPort":80,"ToPort":80,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]},
    {"IpProtocol":"tcp","FromPort":443,"ToPort":443,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]}
  ]' >/dev/null
fi

# IAM instance profile (ECR + SSM)
aws iam create-role --role-name rps-ec2-role --assume-role-policy-document '{
"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
}' >/dev/null 2>&1 || true
aws iam attach-role-policy --role-name rps-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly >/dev/null || true
aws iam attach-role-policy --role-name rps-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore >/dev/null || true
aws iam create-instance-profile --instance-profile-name rps-ec2-profile >/dev/null 2>&1 || true
aws iam add-role-to-instance-profile --instance-profile-name rps-ec2-profile --role-name rps-ec2-role >/dev/null 2>&1 || true
sleep 10

# Key
KEY_NAME=rps-key
aws --region "$REGION" ec2 create-key-pair --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > "${HERE}/../../${KEY_NAME}.pem" 2>/dev/null || true
chmod 400 "${HERE}/../../${KEY_NAME}.pem"

# Instance (idempotent-ish: create if not found)
INSTANCE_ID=$(aws --region "$REGION" ec2 describe-instances --filters "Name=tag:Name,Values=rps-ec2" "Name=instance-state-name,Values=running,pending" --query 'Reservations[0].Instances[0].InstanceId' --output text)
if [ "$INSTANCE_ID" = "None" ]; then
  INSTANCE_ID=$(aws --region "$REGION" ec2 run-instances \
    --image-id "$AMI_ID" --instance-type "${INSTANCE_TYPE:-t3a.large}" \
    --iam-instance-profile Name=rps-ec2-profile \
    --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
    --block-device-mappings DeviceName=/dev/sda1,Ebs={VolumeSize='${VOL_SIZE:-40}',VolumeType=gp3,DeleteOnTermination=true} \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=rps-ec2},{Key=Project,Value=RPS}]" \
    --query 'Instances[0].InstanceId' --output text)
fi
aws --region "$REGION" ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"

# Elastic IP + DNS
ALLOC_ID=$(aws --region "$REGION" ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
PUBLIC_IP=$(aws --region "$REGION" ec2 describe-addresses --allocation-ids "$ALLOC_ID" --query 'Addresses[0].PublicIp' --output text)
aws --region "$REGION" ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$ALLOC_ID" >/dev/null
echo "PUBLIC_IP=$PUBLIC_IP"
