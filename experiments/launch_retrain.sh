#!/bin/bash
# Launch one DeepLPF training run on a GPU instance.
#
# The instance fetches code and data from S3, runs the test suite, trains,
# syncs checkpoints back every ten minutes, and terminates itself on completion
# or at the hard cap. Nothing is hardcoded to one account.
#
# Prerequisites: an S3 bucket holding deeplpf_code.tgz and curl_data.tgz, an
# instance profile with read/write access to it, and a Deep Learning AMI.
#
#   BUCKET=s3://my-bucket AMI=ami-... ./launch_training.sh
set -euo pipefail

: "${BUCKET:?set BUCKET, e.g. s3://curl-train-123456789012}"
: "${AMI:?set AMI, a Deep Learning AMI id for your region}"
INSTANCE_PROFILE="${INSTANCE_PROFILE:-EC2Role}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
REGION="${REGION:-us-east-1}"
SUBNET="${SUBNET:-}"
EPOCHS="${EPOCHS:-1000}"
# Hours. A second, independent guarantee against a hung run billing forever.
HARD_CAP="${HARD_CAP:-48}"
DISK_GB="${DISK_GB:-120}"
EXTRA_FLAGS="${EXTRA_FLAGS:---tf32 --cuda_graphs}"
RUN_NAME="${RUN_NAME:-deeplpf-$(date -u +%Y%m%d-%H%M%S)}"

HERE=$(cd "$(dirname "$0")" && pwd)
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

sed -e "s|__BUCKET__|$BUCKET|g" \
    -e "s|__RUN_NAME__|$RUN_NAME|g" \
    -e "s|__EPOCHS__|$EPOCHS|g" \
    -e "s|__HARD_CAP__|$HARD_CAP|g" \
    -e "s|__EXTRA_FLAGS__|$EXTRA_FLAGS|g" \
    "$HERE/retrain_userdata.sh.tmpl" > "$WORKDIR/userdata.sh"

ARGS=(
  --image-id "$AMI"
  --instance-type "$INSTANCE_TYPE"
  --iam-instance-profile "Name=$INSTANCE_PROFILE"
  --user-data "file://$WORKDIR/userdata.sh"
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,DeleteOnTermination=true}"
  --instance-initiated-shutdown-behavior terminate
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$RUN_NAME},{Key=Project,Value=deeplpf}]"
  --region "$REGION"
)
[ -n "$SUBNET" ] && ARGS+=(--subnet-id "$SUBNET")

ID=$(aws ec2 run-instances "${ARGS[@]}" --query 'Instances[0].InstanceId' --output text)

echo "instance : $ID"
echo "run name : $RUN_NAME"
echo "results  : $BUCKET/runs/$RUN_NAME/"
echo
echo "progress : aws s3 cp $BUCKET/runs/$RUN_NAME/curl-train.log -"
echo "stop it  : aws ec2 terminate-instances --instance-ids $ID --region $REGION"
