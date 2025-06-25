// Simple Memory Chat - Connects to your Memory Garden backend

const API_URL = 'http://159.203.3.40:8000'; // We'll try 8000 first, then 33127

class MemoryChat {
    constructor() {
        this.chatHistory = document.getElementById('chat-history');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.status = document.getElementById('status');
        
        this.init();
    }
    
    async init() {
        // Set up event listeners
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        
        // Test connection and load history
        await this.testConnection();
        await this.loadHistory();
    }
    
    async testConnection() {
        try {
            const response = await fetch(`${API_URL}/health`);
            if (response.ok) {
                this.updateStatus('Connected to Memory Garden', 'success');
            } else {
                throw new Error('Backend not responding');
            }
        } catch (error) {
            // Try alternative port
            this.updateStatus('Trying port 33127...', 'warning');
            // We'll implement fallback in next iteration
        }
    }
    
async sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message) return;
    
    // Display message immediately
    this.displayMessage('You', message);
    this.messageInput.value = '';
    
    try {
        // Send to Memory Garden backend
        const response = await fetch(`${API_URL}/memory/store`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                content: message,
                user_id: "You",
                metadata: {}
            })
        });
        
        if (!response.ok) throw new Error('Failed to save message');
        
        this.updateStatus('Message saved', 'success');
        
        // Get conversation history for context
        const memories = this.chatHistory.innerText || '';

        // Get Claude's response
        const aiResponse = await fetch(`${API_URL}/claude/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: [
                    {role: "user", content: "Previous conversation:\n" + memories + "\n\nNew message: " + message}
                ],
                api_key: "sk-ant-CLAUDE CHANGE HERE"
            })
        });

        const aiData = await aiResponse.json();
        console.log('Claude response:', aiData);
        if (aiData.content && aiData.content[0]) {
            this.displayMessage('Claude', aiData.content[0].text);
        } else if (aiData.error) {
            this.displayMessage('Claude', 'Error: ' + aiData.error.message);
        } else {
            this.displayMessage('Claude', 'Unexpected response format');
        }
        
    } catch (error) {
        this.updateStatus('Error: ' + error.message, 'error');
    }
}
    
    async loadHistory() {
        try {
            // Load previous messages from Memory Garden
            const response = await fetch(`${API_URL}/memory/retrieve`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: "",
                    user_id: "default-user", 
                    include_context: true,
                    max_results: 50
                })
            });
            if (!response.ok) throw new Error('Failed to load history');
            
            const data = await response.json();
            console.log('Full data:', data);
            if (data.memories) {
                data.memories.forEach(msg => {
                    this.displayMessage(msg.user_id || 'Unknown', msg.content);
                });
            }
        } catch (error) {
            this.updateStatus('Could not load history', 'warning');
        }
    }
    
    displayMessage(sender, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message';
        messageDiv.innerHTML = `<strong>${sender}:</strong> ${content}`;
        this.chatHistory.appendChild(messageDiv);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }
    
    updateStatus(message, type = 'info') {
        this.status.textContent = message;
        this.status.className = `status ${type}`;
    }
}

// Start the chat when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MemoryChat();
});