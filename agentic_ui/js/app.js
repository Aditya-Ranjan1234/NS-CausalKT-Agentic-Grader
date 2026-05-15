pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

class AgenticGraderUI {
    constructor() {
        this.uploadedFiles = [];
        this.currentPage = 1;
        this.totalPages = 1;
        this.pdfDoc = null;
        this.analysisResults = null;
        
        this.initElements();
        this.initEventListeners();
    }

    initElements() {
        this.dropzone = document.getElementById('dropzone');
        this.fileInput = document.getElementById('fileInput');
        this.browseBtn = document.getElementById('browseBtn');
        this.uploadedFilesContainer = document.getElementById('uploadedFiles');
        this.uploadActionContainer = document.getElementById('uploadActionContainer');
        this.previewSection = document.getElementById('previewSection');
        this.filePreview = document.getElementById('filePreview');
        this.startAnalysisBtn = document.getElementById('startAnalysisBtn');
        this.systemStatus = document.getElementById('system-status');
        this.uploadSection = document.querySelector('.upload-section');
        this.loadingSection = document.getElementById('loadingSection');
        this.analysisSection = document.getElementById('analysisSection');
        this.imageCanvasContainer = document.getElementById('documentViewer');
        this.pdfCanvas = document.getElementById('pdfCanvas');
        this.prevPageBtn = document.getElementById('prevPage');
        this.nextPageBtn = document.getElementById('nextPage');
        this.pageIndicator = document.getElementById('pageIndicator');
        this.tabButtons = document.querySelectorAll('.tab-btn');
        this.tabContents = document.querySelectorAll('.tab-content');
        this.previewSection = document.getElementById('previewSection');
        this.filePreview = document.getElementById('filePreview');
    }

    initEventListeners() {
        this.browseBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFiles(e.target.files));
        
        this.dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropzone.classList.add('drag-over');
        });
        
        this.dropzone.addEventListener('dragleave', () => {
            this.dropzone.classList.remove('drag-over');
        });
        
        this.dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropzone.classList.remove('drag-over');
            this.handleFiles(e.dataTransfer.files);
        });
        
        this.startAnalysisBtn.addEventListener('click', () => this.startAnalysis());
        
        this.prevPageBtn.addEventListener('click', () => this.changePage(-1));
        this.nextPageBtn.addEventListener('click', () => this.changePage(1));
        
        // Removed tab switching - vertical layout active
    }

    handleFiles(files) {
        Array.from(files).forEach(file => {
            if (['application/pdf', 'image/png', 'image/jpeg'].includes(file.type)) {
                this.uploadedFiles.push(file);
            }
        });
        
        this.renderUploadedFiles();
        this.startAnalysisBtn.disabled = this.uploadedFiles.length === 0;
    }

    renderUploadedFiles() {
        this.uploadedFilesContainer.innerHTML = '';
        
        this.uploadedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            
            const icon = file.type === 'application/pdf' ? '📄' : '🖼️';
            
            fileItem.innerHTML = `
                <div class="file-info">
                    <span class="file-icon">${icon}</span>
                    <div>
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${this.formatFileSize(file.size)}</div>
                    </div>
                </div>
                <button class="remove-btn" data-index="${index}">Remove</button>
            `;
            
            fileItem.querySelector('.remove-btn').addEventListener('click', () => {
                this.removeFile(index);
            });
            
            this.uploadedFilesContainer.appendChild(fileItem);
        });
        
        if (this.uploadedFiles.length > 0) {
            this.showFilePreview(this.uploadedFiles[0]);
        } else {
            this.previewSection.style.display = 'none';
        }
    }

    showFilePreview(file) {
        console.log('Showing preview for file:', file.name, file.type);
        this.uploadActionContainer.style.display = 'block';
        this.previewSection.style.display = 'flex'; // it is inside the container
        this.filePreview.innerHTML = '';
        
        if (file.type.startsWith('image/')) {
            console.log('Rendering image preview');
            const reader = new FileReader();
            reader.onload = (e) => {
                console.log('Image loaded, creating img element');
                const img = document.createElement('img');
                img.src = e.target.result;
                img.style.maxWidth = '100%';
                img.style.maxHeight = '400px';
                img.style.borderRadius = '8px';
                this.filePreview.appendChild(img);
                console.log('Image appended to preview');
            };
            reader.onerror = (error) => {
                console.error('FileReader error:', error);
            };
            reader.readAsDataURL(file);
        } else if (file.type === 'application/pdf') {
            console.log('Rendering PDF placeholder');
            const pdfPlaceholder = document.createElement('div');
            pdfPlaceholder.innerHTML = `
                <div style="text-align: center; padding: 40px; background: #0f172a; border-radius: 8px;">
                    <p style="color: #cbd5e1; font-size: 24px;">📄 PDF Uploaded: ${file.name}</p>
                    <p style="color: #94a3b8; margin-top: 12px;">Preview will be shown during analysis</p>
                </div>
            `;
            this.filePreview.appendChild(pdfPlaceholder);
        }
    }

    removeFile(index) {
        this.uploadedFiles.splice(index, 1);
        this.renderUploadedFiles();
        this.startAnalysisBtn.disabled = this.uploadedFiles.length === 0;
    }

    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    async startAnalysis() {
        this.uploadSection.style.display = 'none';
        this.loadingSection.style.setProperty('display', 'flex', 'important');
        this.systemStatus.textContent = '● Processing...';
        this.systemStatus.className = 'status-indicator processing';
        
        this.simulateLoadingSteps();
        
        try {
            const results = await this.sendToBackend();
            this.analysisResults = results;
            this.showAnalysis();
        } catch (error) {
            console.error('Analysis failed:', error);
            alert('Analysis failed. Please try again.');
            this.resetUI();
        }
    }

    simulateLoadingSteps() {
        const loadingMessages = [
            "Initializing...",
            "Processing files...",
            "Running NS-CausalKT...",
            "Analyzing response...",
            "Generating feedback..."
        ];
        
        let currentStep = 0;
        const loadingStatusEl = document.getElementById('loadingStatus');
        
        const interval = setInterval(() => {
            if (loadingStatusEl && currentStep < loadingMessages.length) {
                loadingStatusEl.textContent = loadingMessages[currentStep];
                currentStep++;
            } else if (currentStep >= loadingMessages.length) {
                clearInterval(interval);
            }
        }, 800);
    }

    async sendToBackend() {
        const formData = new FormData();
        this.uploadedFiles.forEach(file => {
            formData.append('files', file);
        });
        
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Backend error');
        }
        
        return await response.json();
    }

    async showAnalysis() {
        this.loadingSection.style.display = 'none';
        this.analysisSection.style.setProperty('display', 'grid', 'important');
        this.systemStatus.textContent = '● Analysis Complete';
        this.systemStatus.className = 'status-indicator ready';
        
        if (this.uploadedFiles[0].type === 'application/pdf') {
            await this.loadPDF(this.uploadedFiles[0]);
        } else {
            this.loadImage(this.uploadedFiles[0]);
        }
        
        this.populateFeedback();
    }

    async loadPDF(file) {
        const arrayBuffer = await file.arrayBuffer();
        this.pdfDoc = await pdfjsLib.getDocument(arrayBuffer).promise;
        this.totalPages = this.pdfDoc.numPages;
        this.currentPage = 1;
        this.updatePageIndicator();
        this.renderPage(this.currentPage);
    }

    async renderPage(pageNum) {
        const page = await this.pdfDoc.getPage(pageNum);
        const scale = 1.5;
        const viewport = page.getViewport({ scale });
        
        const canvas = this.pdfCanvas;
        const context = canvas.getContext('2d');
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        
        const renderContext = {
            canvasContext: context,
            viewport: viewport
        };
        
        await page.render(renderContext).promise;
    }

    loadImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const canvas = this.pdfCanvas;
                const ctx = canvas.getContext('2d');
                const containerWidth = 800;
                const containerHeight = 600;
                
                let drawWidth, drawHeight, offsetX, offsetY;
                const imgRatio = img.width / img.height;
                const containerRatio = containerWidth / containerHeight;
                
                if (imgRatio > containerRatio) {
                    drawWidth = containerWidth;
                    drawHeight = containerWidth / imgRatio;
                    offsetX = 0;
                    offsetY = (containerHeight - drawHeight) / 2;
                } else {
                    drawHeight = containerHeight;
                    drawWidth = containerHeight * imgRatio;
                    offsetX = (containerWidth - drawWidth) / 2;
                    offsetY = 0;
                }
                
                canvas.width = containerWidth;
                canvas.height = containerHeight;
                ctx.fillStyle = '#020617';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    changePage(delta) {
        const newPage = this.currentPage + delta;
        if (newPage >= 1 && newPage <= this.totalPages) {
            this.currentPage = newPage;
            this.updatePageIndicator();
            this.renderPage(this.currentPage);
        }
    }

    updatePageIndicator() {
        this.pageIndicator.textContent = `${this.currentPage} / ${this.totalPages}`;
        this.prevPageBtn.disabled = this.currentPage === 1;
        this.nextPageBtn.disabled = this.currentPage === this.totalPages;
    }

    populateFeedback() {
        if (!this.analysisResults) return;
        
        const data = this.analysisResults;
        
        const setField = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value || '--';
        };

        setField('overallScore', data.overall_score);
        const kt = data.ns_causalkt || {};
        let summary = data.summary || 'Analysis summary will appear here...';
        if (kt.active && kt.prediction) {
            const probability = Math.round((kt.prediction.passing_probability || 0) * 100);
            const target = kt.prediction.target_skill || 'target concept';
            const weak = (kt.weakest_concepts || [])
                .slice(0, 3)
                .map(item => `${item.skill}: ${Math.round((item.mastery || 0) * 100)}%`)
                .join(', ');
            summary += `\n\nNS-CausalKT prediction for ${target}: ${probability}% pass probability.`;
            if (weak) summary += ` Weakest mapped concepts: ${weak}.`;
            if (kt.counterfactual) {
                const after = Math.round((kt.counterfactual.intervened_probability || 0) * 100);
                summary += ` Counterfactual do(${kt.counterfactual.intervened_concept}=100%) -> ${after}% pass probability.`;
            }
        } else if (kt.error) {
            summary += `\n\nNS-CausalKT could not run: ${kt.error}`;
        }
        setField('summaryText', summary);
        
        const modelStatus = document.getElementById('modelStatus');
        if (modelStatus) {
            if (kt.active && kt.prediction) {
                const probability = Math.round((kt.prediction.passing_probability || 0) * 100);
                modelStatus.textContent = `KT-Model: Active (${probability}% predicted)`;
                modelStatus.classList.remove('inactive');
            } else if (data.kt_active) {
                modelStatus.textContent = 'KT-Model: Math Detected, No Model Output';
                modelStatus.classList.add('inactive');
            } else {
                modelStatus.textContent = 'KT-Model: Inactive (Non-Math Topic)';
                modelStatus.classList.add('inactive');
            }
        }
        
        this.renderFeedbackList('mistakesList', data.mistakes || [], 'error');
        this.renderFeedbackList('correctionsList', data.corrections || [], 'success');
        this.renderFeedbackList('strengthsList', data.strengths || [], 'success');
        this.renderFeedbackList('weaknessesList', data.weaknesses || [], 'warning');
        this.renderFeedbackList('focusList', data.focus_areas || [], 'warning');
        
    }

    renderFeedbackList(containerId, items, type) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        items.forEach(item => {
            const el = document.createElement('div');
            el.className = `feedback-item ${type}`;
            el.innerHTML = `
                <div class="feedback-item-header">
                    <span class="feedback-item-title">${item.title}</span>
                    <span class="feedback-item-tag tag-${type}">Q${item.question || 'N/A'}</span>
                </div>
                <p class="feedback-item-desc">${item.description}</p>
                ${item.correction ? `<div class="feedback-item-correction">${item.correction}</div>` : ''}
            `;
            container.appendChild(el);
        });
    }

    resetUI() {
        // Remove tab event listeners
        // Tabs are now replaced by vertical scroll sections
        this.uploadSection.style.display = 'block';
        this.loadingSection.style.display = 'none';
        this.analysisSection.style.display = 'none';
        this.systemStatus.textContent = '● System Ready';
        this.systemStatus.className = 'status-indicator ready';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.agenticGrader = new AgenticGraderUI();
});
