custom_js = """
<style>
.context-menu {
    position: absolute;
    background: white;
    border: 1px solid #ccc;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    padding: 12px;
    z-index: 10000;
    min-width: 280px;
    max-width: 400px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.4;
}
.context-menu h4 {
    margin: 0 0 8px 0;
    color: #2c3e50;
    font-size: 16px;
}
.context-menu .section {
    margin-bottom: 10px;
}
.context-menu .section-title {
    font-weight: bold;
    color: #34495e;
    margin-bottom: 4px;
    font-size: 13px;
}
.context-menu ul {
    padding-left: 16px;
    margin: 4px 0;
}
.context-menu li {
    margin-bottom: 3px;
}
.context-menu a {
    color: #2980b9;
    text-decoration: none;
}
.context-menu a:hover {
    text-decoration: underline;
}
.context-menu button {
    width: 100%;
    padding: 7px;
    margin-top: 8px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
}
.context-menu .delete-btn {
    background: #e74c3c;
    color: white;
}
.context-menu .delete-btn:hover {
    background: #c0392b;
}
.context-menu .close-btn {
    background: #95a5a6;
    color: white;
    margin-top: 4px;
}
.context-menu .close-btn:hover {
    background: #7f8c8d;
}
</style>

<script type="text/javascript">
document.addEventListener('click', function(e) {
    const menu = document.getElementById('contextMenu');
    if (menu && !menu.contains(e.target)) {
        menu.remove();
    }
});

network.on("click", function(params) {
    const existingMenu = document.getElementById('contextMenu');
    if (existingMenu) existingMenu.remove();

    if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        fetch('/get-node/' + encodeURIComponent(nodeId))
            .then(response => response.json())
            .then(node => {
                if (node.error) return;

                const menu = document.createElement('div');
                menu.id = 'contextMenu';
                menu.className = 'context-menu';
                menu.style.left = (params.pointer.DOM.x + 5) + 'px';
                menu.style.top = (params.pointer.DOM.y + 5) + 'px';

                // Format titles
                const titlesHtml = node.title.map(t => `<li>${t}</li>`).join('');
                // Format URLs
                const urlsHtml = node.url.map(u => {
                    const cleanUrl = u.trim();
                    return `<li><a href="${cleanUrl}" target="_blank">${cleanUrl}</a></li>`;
                }).join('');

                menu.innerHTML = `
                    <h4>${nodeId}</h4>
                    <div class="section">
                        <div class="section-title">Type:</div>
                        <div>${node.type || '—'}</div>
                    </div>
                    <div class="section">
                        <div class="section-title">Titles:</div>
                        <ul>${titlesHtml}</ul>
                    </div>
                    <div class="section">
                        <div class="section-title">URLs:</div>
                        <ul>${urlsHtml}</ul>
                    </div>
                    <button class="delete-btn" onclick="deleteNode('${nodeId}')">Delete Node</button>
                    <button class="close-btn" onclick="this.parentElement.remove()">Close</button>
                `;
                document.body.appendChild(menu);
            })
            .catch(err => console.error('Failed to load node:', err));
    }
});

function deleteNode(nodeId) {
    if (!confirm('Delete node "' + nodeId + '" and all its connections?')) return;
    
    fetch('/delete-node/' + encodeURIComponent(nodeId), { method: 'DELETE' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                location.reload();
            } else {
                alert('Error: ' + (data.error || 'Unknown'));
            }
        });
}
</script>"""