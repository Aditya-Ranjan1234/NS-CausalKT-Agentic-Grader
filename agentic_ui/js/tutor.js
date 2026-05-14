document.addEventListener('DOMContentLoaded', () => {
    initTutor();
});

let currentUser = null;
let currentUserId = 'aditya_ranjan';

async function initTutor() {
    await fetchUsers();
    await fetchProfile();
    await fetchGraph();
    
    document.getElementById('sendBtn').addEventListener('click', sendMessage);
    document.getElementById('chatInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
}

async function fetchUsers() {
    try {
        const response = await fetch('/api/tutor/users');
        if (response.ok) {
            const users = await response.json();
            const select = document.getElementById('userSelect');
            select.innerHTML = '';
            users.forEach(user => {
                const option = document.createElement('option');
                option.value = user.id;
                option.textContent = user.name;
                if (user.id === currentUserId) option.selected = true;
                select.appendChild(option);
            });
            select.addEventListener('change', async (e) => {
                currentUserId = e.target.value;
                await fetchProfile();
                await fetchGraph();
            });
        }
    } catch (e) {
        console.error("Failed to load users", e);
    }
}

async function fetchProfile() {
    try {
        const response = await fetch(`/api/tutor/profile?user_id=${currentUserId}`);
        if (response.ok) {
            currentUser = await response.json();
            updateProfileUI();
        }
    } catch (e) {
        console.error("Failed to load profile", e);
    }
}

function updateProfileUI() {
    if (!currentUser) return;
    
    document.getElementById('userName').textContent = currentUser.name;
    document.getElementById('currentGoal').textContent = currentUser.learning_goals[0] || 'No goal set';
    
    const masterList = document.getElementById('masteredList');
    masterList.innerHTML = '';
    currentUser.concepts_learned.forEach(c => {
        const li = document.createElement('li');
        li.textContent = c;
        masterList.appendChild(li);
    });
    
    const weakList = document.getElementById('weakList');
    weakList.innerHTML = '';
    currentUser.weak_areas.forEach(c => {
        const li = document.createElement('li');
        li.textContent = c;
        weakList.appendChild(li);
    });
    
    // Render chat history
    const chatHistory = document.getElementById('chatHistory');
    // Clear everything except the first intro message
    while(chatHistory.children.length > 1) {
        chatHistory.removeChild(chatHistory.lastChild);
    }
    
    currentUser.interaction_history.forEach(interaction => {
        appendMessage('user', interaction.user);
        appendMessage('bot', interaction.bot);
    });
}

async function fetchGraph() {
    try {
        const response = await fetch('/api/tutor/graph');
        if (response.ok) {
            const data = await response.json();
            document.getElementById('recommendedNext').textContent = data.recommended_next;
            drawDAG(data.nodes, data.edges);
        }
    } catch (e) {
        console.error("Failed to load graph", e);
    }
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (!msg) return;
    
    input.value = '';
    appendMessage('user', msg);
    
    // Show a temporary loading message
    const tempId = appendMessage('bot', 'Thinking...');
    
    try {
        const response = await fetch('/api/tutor/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg})
        });
        
        const data = await response.json();
        
        // Remove temp message
        const tempEl = document.getElementById(tempId);
        if (tempEl) tempEl.remove();
        
        appendMessage('bot', data.reply);
        
        if (data.updated_mastery) {
            await fetchProfile(); // Refresh profile if mastery changed
            await fetchGraph(); // Refresh graph
        }
    } catch (e) {
        console.error("Chat failed", e);
        const tempEl = document.getElementById(tempId);
        if (tempEl) tempEl.textContent = "Error: Could not reach the AI Tutor.";
    }
}

function appendMessage(sender, text) {
    const chatHistory = document.getElementById('chatHistory');
    const msgDiv = document.createElement('div');
    const id = 'msg_' + Date.now();
    msgDiv.id = id;
    msgDiv.className = `chat-message message-${sender}`;
    msgDiv.textContent = text;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return id;
}

// D3.js DAG Visualization
function drawDAG(nodes, edges) {
    const container = document.getElementById('dagContainer');
    container.innerHTML = '';
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    const svg = d3.select("#dagContainer").append("svg")
        .attr("width", width)
        .attr("height", height);
        
    // Define arrow markers for graph links
    svg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('xoverflow', 'visible')
        .append('svg:path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#334155')
        .style('stroke','none');

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(edges).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide().radius(40));

    const link = svg.append("g")
        .selectAll("line")
        .data(edges)
        .enter().append("line")
        .attr("class", "link")
        .attr('marker-end','url(#arrowhead)');

    const node = svg.append("g")
        .selectAll("g")
        .data(nodes)
        .enter().append("g")
        .attr("class", d => `node ${d.status}`)
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("circle")
        .attr("r", 15);

    node.append("text")
        .attr("dy", 25)
        .text(d => d.id);

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}
