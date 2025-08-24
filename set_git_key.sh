#!/bin/bash
cp /workspace/credentials/jjc.chris.outlook ~/.ssh
git config core.sshCommand 'ssh -i ~/.ssh/jjc.chris.outlook'
