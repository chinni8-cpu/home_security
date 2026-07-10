// ============================================
// DASHBOARD SCRIPT WITH ENHANCED FEATURES
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ Dashboard loaded');
    
    // Load data
    loadFamilyMembers();
    loadAlerts();
    
    // Auto-refresh alerts every 5 seconds
    setInterval(loadAlerts, 5000);
    setInterval(updateStats, 10000);
    
    // ============ START CAMERA ============
    document.getElementById('startCamera').addEventListener('click', function() {
        const btn = this;
        btn.textContent = '⏳ Starting...';
        btn.disabled = true;
        
        fetch('/start_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const videoFeed = document.getElementById('video-feed');
                videoFeed.src = '/video_feed?' + new Date().getTime();
                videoFeed.style.display = 'block';
                
                // Update status
                document.getElementById('statusDot').className = 'status-dot active';
                document.getElementById('camStatus').textContent = '🟢 Online';
                document.getElementById('camStatus').style.color = '#2ecc71';
                
                showToast('Camera started successfully!', 'success');
            } else {
                showToast('Error: ' + data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error starting camera. Make sure your webcam is connected.', 'error');
        })
        .finally(() => {
            btn.textContent = '▶ Start Camera';
            btn.disabled = false;
        });
    });
    
    // ============ STOP CAMERA ============
    document.getElementById('stopCamera').addEventListener('click', function() {
        const btn = this;
        btn.textContent = '⏳ Stopping...';
        btn.disabled = true;
        
        fetch('/stop_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const videoFeed = document.getElementById('video-feed');
                videoFeed.src = '';
                
                // Update status
                document.getElementById('statusDot').className = 'status-dot inactive';
                document.getElementById('camStatus').textContent = '🔴 Offline';
                document.getElementById('camStatus').style.color = '#e74c3c';
                
                showToast('Camera stopped successfully!', 'info');
            } else {
                showToast('Error: ' + data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error stopping camera.', 'error');
        })
        .finally(() => {
            btn.textContent = '⏹ Stop Camera';
            btn.disabled = false;
        });
    });
    
    // ============ ADD FAMILY MEMBER ============
    document.getElementById('addMemberForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const name = document.getElementById('memberName').value.trim();
        const fileInput = document.getElementById('memberImage');
        
        if (!name) {
            showToast('Please enter a name', 'error');
            return;
        }
        
        if (!fileInput.files || fileInput.files.length === 0) {
            showToast('Please select an image', 'error');
            return;
        }
        
        const formData = new FormData();
        formData.append('name', name);
        formData.append('image', fileInput.files[0]);
        
        const submitBtn = this.querySelector('.btn-add');
        const originalText = submitBtn.textContent;
        submitBtn.textContent = '⏳ Adding...';
        submitBtn.disabled = true;
        
        fetch('/add_family_member', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast(`✅ ${name} added successfully!`, 'success');
                loadFamilyMembers();
                this.reset();
            } else {
                showToast('Error: ' + (data.error || 'Unknown error'), 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error adding family member.', 'error');
        })
        .finally(() => {
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
        });
    });
    
    // ============ WHATSAPP ============
    document.getElementById('whatsappBtn').addEventListener('click', function() {
        const btn = this;
        btn.textContent = '⏳ Opening...';
        btn.disabled = true;
        
        fetch('/send_whatsapp', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.whatsapp_url) {
                window.open(data.whatsapp_url, '_blank');
                showToast('📱 WhatsApp opened!', 'success');
            } else {
                showToast('Error: ' + (data.error || 'Could not generate link'), 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error opening WhatsApp', 'error');
        })
        .finally(() => {
            btn.textContent = '📱 WhatsApp';
            btn.disabled = false;
        });
    });
    
    // ============ EMAIL ============
    document.getElementById('emailBtn').addEventListener('click', function() {
        const btn = this;
        btn.textContent = '⏳ Opening...';
        btn.disabled = true;
        
        fetch('/send_email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.email_url) {
                window.open(data.email_url, '_blank');
                showToast('📧 Email client opened!', 'success');
            } else {
                showToast('Error: ' + (data.error || 'Could not generate link'), 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error opening email client', 'error');
        })
        .finally(() => {
            btn.textContent = '📧 Email';
            btn.disabled = false;
        });
    });
});

// ============================================
// LOAD FAMILY MEMBERS
// ============================================
function loadFamilyMembers() {
    fetch('/get_family_members')
        .then(response => response.json())
        .then(data => {
            const list = document.getElementById('membersList');
            list.innerHTML = '';
            
            // Update count
            document.getElementById('memberCount').textContent = data.length;
            
            if (data.length === 0) {
                list.innerHTML = `
                    <div style="text-align: center; padding: 30px 10px; color: rgba(255,255,255,0.3); grid-column: 1 / -1;">
                        <span style="font-size: 40px; display: block; margin-bottom: 10px;">👤</span>
                        No family members added yet
                    </div>
                `;
                return;
            }
            
            data.forEach(member => {
                const div = document.createElement('div');
                div.className = 'member-item';
                
                let imageSrc = member.image_path || '';
                if (imageSrc && !imageSrc.startsWith('http') && !imageSrc.startsWith('/')) {
                    imageSrc = '/' + imageSrc;
                }
                
                div.innerHTML = `
                    <img src="${imageSrc}" alt="${member.name}" 
                         onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2270%22 height=%2270%22%3E%3Crect fill=%22%232a2a4a%22 width=%2270%22 height=%2270%22/%3E%3Ctext x=%2235%22 y=%2242%22 font-size=%2230%22 text-anchor=%22middle%22 fill=%22%23a8b2d1%22%3E👤%3C/text%3E%3C/svg%3E'">
                    <span>${member.name}</span>
                    <button class="delete-btn" onclick="deleteMember(${member.id}, '${member.name}')">×</button>
                `;
                list.appendChild(div);
            });
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('membersList').innerHTML = 
                '<p style="color: #e74c3c; text-align: center; padding: 20px;">❌ Error loading members</p>';
        });
}

// ============================================
// DELETE FAMILY MEMBER
// ============================================
function deleteMember(id, name) {
    if (confirm(`⚠️ Are you sure you want to delete "${name}"?\n\nThis will remove them from the system.`)) {
        fetch(`/delete_family_member/${id}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast(`✅ ${name} deleted successfully!`, 'success');
                loadFamilyMembers();
            } else {
                showToast('Error: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error deleting member', 'error');
        });
    }
}

// ============================================
// LOAD ALERTS
// ============================================
function loadAlerts() {
    fetch('/get_alerts')
        .then(response => response.json())
        .then(data => {
            const list = document.getElementById('alertsList');
            list.innerHTML = '';
            
            // Update count
            document.getElementById('alertCount').textContent = data.length;
            
            if (data.length === 0) {
                list.innerHTML = `
                    <div style="text-align: center; padding: 30px 10px; color: rgba(255,255,255,0.3);">
                        <span style="font-size: 40px; display: block; margin-bottom: 10px;">✅</span>
                        No alerts yet
                    </div>
                `;
                return;
            }
            
            data.forEach((alert, index) => {
                const div = document.createElement('div');
                div.className = 'alert-item';
                div.style.animationDelay = (index * 0.1) + 's';
                div.innerHTML = `
                    <div>⚠️ ${alert.message}</div>
                    <div class="time">${alert.created_at}</div>
                `;
                list.appendChild(div);
            });
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// ============================================
// UPDATE STATS
// ============================================
function updateStats() {
    // Just reload counts
    loadFamilyMembers();
}

// ============================================
// TOAST NOTIFICATION
// ============================================
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    const icon = toast.querySelector('.toast-icon');
    
    toastMessage.textContent = message;
    toast.className = 'toast ' + type;
    
    // Set icon
    if (type === 'success') icon.textContent = '✅';
    else if (type === 'error') icon.textContent = '❌';
    else if (type === 'info') icon.textContent = 'ℹ️';
    
    toast.classList.add('show');
    
    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => {
        toast.classList.remove('show');
    }, 4000);
}

// ============================================
// VIDEO FEED ERROR HANDLING
// ============================================
window.addEventListener('error', function(e) {
    if (e.target.tagName === 'IMG' && e.target.id === 'video-feed') {
        console.log('Video feed error - camera may be stopped');
        // Update status to inactive
        document.getElementById('statusDot').className = 'status-dot inactive';
        document.getElementById('camStatus').textContent = '🔴 Offline';
        document.getElementById('camStatus').style.color = '#e74c3c';
    }
});

console.log('✅ Script loaded successfully');