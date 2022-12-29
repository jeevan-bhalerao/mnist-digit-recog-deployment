## create image
docker build -t my-fastapi-image .

## create container
docker run -d --name my-fastapi-container -p 80:8001 my-fastapi-image

## doc
https://fastapi.tiangolo.com/deployment/docker/


