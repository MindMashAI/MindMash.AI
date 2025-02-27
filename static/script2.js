document.addEventListener("DOMContentLoaded", () => {
    const socket = io();
    const chatBox = document.getElementById("chatBox");
    const messageInput = document.getElementById("messageInput");
    const sendChat = document.getElementById("sendChat");
    const turnsInput = document.getElementById("turnsInput");
    const aiSelect = document.getElementById("aiSelect");
    const loading = document.getElementById("loading");
    const progress = document.getElementById("progress");
    // const sendSound = new Audio('/static/audio/send.wav');
    // const replySound = new Audio('/static/audio/reply.wav');
    let messageCount = 0;
    const displayName = "{{ display_name if display_name else username | safe }}" || "Q"; // Match your logs

    console.log("Script loaded, displayName:", displayName);

    function addMessage(speaker, content, isTyping = false) {
        console.log("Adding message:", speaker, content);
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${speaker.toLowerCase().replace(/\s/g, '-')}`;
        messageDiv.innerHTML = `<span class="speaker">${speaker}:</span> <span class="message-content">${content}</span>`;
        if (isTyping) {
            messageDiv.dataset.typing = "true"; // Mark as typing
            messageDiv.style.opacity = "0.7"; // Subtle fade for typing
        } else {
            messageDiv.style.opacity = "1";
        }
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight; // Ensure scroll to bottom
    }

    function removeTypingMessages() {
        const typingMessages = chatBox.querySelectorAll('.chat-message.system[data-typing="true"]');
        typingMessages.forEach(msg => msg.remove());
    }

    socket.on('chat_update', (msg) => {
        console.log("Received chat_update:", msg);
        if (msg.speaker === "System" && msg.content.includes("is typing...")) {
            addMessage(msg.speaker, msg.content, true);
        } else {
            removeTypingMessages(); // Clear all typing messages before adding response
            addMessage(msg.speaker, msg.content);
            if (msg.speaker !== displayName) {
                // replySound.play();
                messageCount++;
                const progressWidth = Math.min((messageCount / 20) * 100, 100);
                progress.style.width = `${progressWidth}%`;
            }
        }
    });

    function sendChatMessage() {
        const message = messageInput.value.trim();
        const turns = parseInt(turnsInput.value) || 1;
        const selectedAI = aiSelect.value;
        if (!message) return;
        socket.emit('chat_message', { message, turns, selected_ai: selectedAI });
        // sendSound.play();
        messageInput.value = "";
        sendChat.disabled = true;
        loading.style.display = "block";
        setTimeout(() => {
            sendChat.disabled = false;
            loading.style.display = "none";
        }, 1000);
    }

    sendChat.addEventListener("click", sendChatMessage);
    messageInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendChatMessage();
    });

    messageInput.focus();
});