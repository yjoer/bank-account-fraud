#!/bin/bash

if [ "$1" == "swap" ]; then
  ansible-playbook -i inventory.yml swap.yml
fi

if [ "$1" == "docker" ]; then
  ansible-playbook -i inventory.yml docker.yml
fi
