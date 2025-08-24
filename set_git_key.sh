#!/bin/bash
cp /workspace/credentials/jjc.chris.outlook ~/.ssh
git config core.sshCommand 'ssh -i ~/.ssh/jjc.chris.outlook'
chmod 600 ~/.ssh/jjc.chris.outlook

git config --global user.email "junjiechen.chris@outlook.com"
git config --global user.name "Junjie Chen"
