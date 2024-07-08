#!/bin/bash

docker login 
docker tag sarlib:latest sirbastiano94/sarlib:latest
docker push sirbastiano94/sarlib:latest  