#!/bin/bash

set -e

ruff check --ignore=E501,E402 $@ .
python3 -m nbqa ruff check --ignore=E501,E402 $@ .
