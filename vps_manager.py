#!/usr/bin/env python3
"""
VPS Manager - Your hands-on VPS management tool
You're the meatsack with fingers, this is your tool!
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import paramiko
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

console = Console()

class VPSManager:
    def __init__(self):
        self.config_file = Path("vps_config.json")
        self.servers = self.load_config()
        
    def load_config(self) -> Dict:
        """Load server configuration from JSON file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_config(self):
        """Save server configuration to JSON file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.servers, f, indent=2)
    
    def add_server(self, name: str, ip: str, username: str, key_path: str = None):
        """Add a new server to the configuration"""
        server_info = {
            "ip": ip,
            "username": username,
            "key_path": key_path,
            "status": "unknown"
        }
        self.servers[name] = server_info
        self.save_config()
        console.print(f"[green]✓[/green] Server '{name}' added successfully!")
    
    def list_servers(self):
        """Display all configured servers"""
        if not self.servers:
            console.print("[yellow]No servers configured yet.[/yellow]")
            return
        
        table = Table(title="Configured VPS Servers")
        table.add_column("Name", style="cyan")
        table.add_column("IP Address", style="magenta")
        table.add_column("Username", style="green")
        table.add_column("Status", style="yellow")
        
        for name, info in self.servers.items():
            table.add_row(name, info["ip"], info["username"], info["status"])
        
        console.print(table)
    
    def connect_server(self, name: str):
        """Connect to a server via SSH"""
        if name not in self.servers:
            console.print(f"[red]Error: Server '{name}' not found![/red]")
            return
        
        server = self.servers[name]
        console.print(f"[blue]Connecting to {name} ({server['ip']})...[/blue]")
        
        try:
            # This is where you'd implement SSH connection
            # For now, we'll just show the connection info
            console.print(f"[green]Would connect to:[/green]")
            console.print(f"  Host: {server['ip']}")
            console.print(f"  User: {server['username']}")
            if server['key_path']:
                console.print(f"  Key: {server['key_path']}")
            
            # In a real implementation, you'd use paramiko here
            # ssh = paramiko.SSHClient()
            # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # ssh.connect(server['ip'], username=server['username'], key_filename=server['key_path'])
            
        except Exception as e:
            console.print(f"[red]Connection failed: {e}[/red]")
    
    def run_command(self, name: str, command: str):
        """Run a command on a server"""
        if name not in self.servers:
            console.print(f"[red]Error: Server '{name}' not found![/red]")
            return
        
        server = self.servers[name]
        console.print(f"[blue]Running '{command}' on {name}...[/blue]")
        
        # This is where you'd implement remote command execution
        # For now, we'll simulate it
        console.print(f"[yellow]Simulated output from {server['ip']}:[/yellow]")
        console.print(f"$ {command}")
        console.print("Command would be executed here...")

@click.group()
def cli():
    """VPS Manager - Your hands-on server management tool"""
    pass

@cli.command()
@click.option('--name', prompt='Server name', help='Name for this server')
@click.option('--ip', prompt='IP address', help='Server IP address')
@click.option('--username', prompt='Username', help='SSH username')
@click.option('--key-path', help='Path to SSH private key')
def add(name, ip, username, key_path):
    """Add a new server to your configuration"""
    manager = VPSManager()
    manager.add_server(name, ip, username, key_path)

@cli.command()
def list():
    """List all configured servers"""
    manager = VPSManager()
    manager.list_servers()

@cli.command()
@click.argument('name')
def connect(name):
    """Connect to a server via SSH"""
    manager = VPSManager()
    manager.connect_server(name)

@cli.command()
@click.argument('name')
@click.argument('command')
def run(name, command):
    """Run a command on a server"""
    manager = VPSManager()
    manager.run_command(name, command)

@cli.command()
def setup():
    """Initial setup and welcome message"""
    console.print(Panel.fit(
        "[bold blue]Welcome to VPS Manager![/bold blue]\n\n"
        "You're the meatsack with fingers, and this is your tool!\n\n"
        "Available commands:\n"
        "• vps_manager.py add - Add a new server\n"
        "• vps_manager.py list - List all servers\n"
        "• vps_manager.py connect <server> - Connect to a server\n"
        "• vps_manager.py run <server> <command> - Run a command\n\n"
        "Let's get started!",
        title="VPS Manager Setup"
    ))

if __name__ == "__main__":
    cli() 