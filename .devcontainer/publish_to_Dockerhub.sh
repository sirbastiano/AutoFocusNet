#!/bin/bash

# Variables
USERNAME="sirbastiano94"
IMAGE_NAME="sarlib"
TAG="latest"
COMPOSE_FILE="docker-compose.yml"

# Functions
docker_login() {
    echo "Logging in to Docker Hub..."
    docker login || { echo "Docker login failed"; exit 1; }
}

build_images() {
    echo "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build || { echo "Docker build failed"; exit 1; }
}

tag_images() {
    echo "Tagging Docker images..."
    services=$(docker-compose -f $COMPOSE_FILE config --services)
    for service in $services; do
        local_image=$(docker-compose -f $COMPOSE_FILE config | grep "image:" | awk '{print $2}')
        docker tag $local_image $USERNAME/$IMAGE_NAME:$TAG || { echo "Docker tag failed for $local_image"; exit 1; }
    done
}

push_images() {
    echo "Pushing Docker images to Docker Hub..."
    services=$(docker-compose -f $COMPOSE_FILE config --services)
    for service in $services; do
        docker push $USERNAME/$IMAGE_NAME:$TAG || { echo "Docker push failed"; exit 1; }
    done
}

# Script Execution
docker_login
build_images
tag_images
push_images

echo "Docker Compose project published to Docker Hub successfully!"