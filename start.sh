#!/bin/bash
set -e

sudo xhost +local:
xhost +local:docker
docker compose up --build
