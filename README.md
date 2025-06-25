# VPS Manager - Your Hands-On Server Management Tool

Welcome, meatsack with fingers! This is your Python-based VPS management tool for spinning up and managing virtual private servers.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the setup:**
   ```bash
   python vps_manager.py setup
   ```

3. **Add your first server:**
   ```bash
   python vps_manager.py add
   ```

## 📋 Available Commands

### Add a Server
```bash
python vps_manager.py add --name my-server --ip 192.168.1.100 --username root --key-path ~/.ssh/id_rsa
```

### List All Servers
```bash
python vps_manager.py list
```

### Connect to a Server
```bash
python vps_manager.py connect my-server
```

### Run Commands on a Server
```bash
python vps_manager.py run my-server "ls -la"
```

## 🛠️ Features

- **Server Management**: Add, list, and manage multiple VPS servers
- **SSH Integration**: Connect to servers via SSH (using paramiko)
- **Remote Commands**: Execute commands on remote servers
- **Rich CLI**: Beautiful terminal interface with colors and tables
- **Configuration**: Persistent server configuration in JSON format

## 🔧 Configuration

The tool stores server configurations in `vps_config.json`:
```json
{
  "my-server": {
    "ip": "192.168.1.100",
    "username": "root",
    "key_path": "~/.ssh/id_rsa",
    "status": "unknown"
  }
}
```

## 🎯 Next Steps

This is a foundation - you can extend it with:
- VPS provider APIs (DigitalOcean, AWS, etc.)
- Automated server provisioning
- Monitoring and health checks
- Backup and restore functionality
- Multi-server orchestration

## 🐛 Troubleshooting

- Make sure your SSH keys are properly configured
- Check that your VPS is accessible from your network
- Verify Python dependencies are installed correctly

Remember: You're the meatsack with fingers, this tool is just here to help! 🖐️ 