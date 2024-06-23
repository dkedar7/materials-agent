gcloud config set project web-apps-273916
IMAGE_URL="us-east1-docker.pkg.dev/web-apps-273916/cloud-run-source-deploy/materials-agent:latest"
docker build . --tag $IMAGE_URL
docker push $IMAGE_URL
gcloud run deploy materials-agent --image $IMAGE_URL --platform managed --region us-east1