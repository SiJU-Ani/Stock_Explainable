// Global state
let pipelineRunning = false;

// Step definitions
const STEPS = [
    { id: 'step1', name: 'Data Acquisition', detail: 'Fetching stock data from yfinance...' },
    { id: 'step2', name: 'Graph Construction', detail: 'Building knowledge graph...' },
    { id: 'step3', name: 'GNN Processing', detail: 'Running Graph Neural Network...' },
    { id: 'step4', name: 'Temporal Modeling', detail: 'LSTM temporal prediction...' },
    { id: 'step5', name: 'Evaluation', detail: 'Computing financial metrics...' }
];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    const runButton = document.getElementById('runBtn');
    if (runButton) {
        runButton.addEventListener('click', runPipeline);
    }
});

// Main pipeline execution function
async function runPipeline() {
    if (pipelineRunning) return;
    
    const runButton = document.getElementById('runBtn');
    const statusText = document.getElementById('statusText');
    
    pipelineRunning = true;
    runButton.disabled = true;
    statusText.textContent = '⏳ Running...';
    
    // Reset all steps
    STEPS.forEach(step => {
        resetStepCard(step.id);
    });
    
    // Clear metrics
    clearMetrics();
    
    try {
        // Simulate step progress
        for (let i = 0; i < STEPS.length; i++) {
            const step = STEPS[i];
            updateStepProgress(step.id, 'pending');
            
            // Wait for step duration
            await sleep(700);
            updateStepProgress(step.id, 'progress');
            await sleep(300);
        }
        
        // Fetch actual pipeline results
        const response = await fetch('/api/run-pipeline');
        const data = await response.json();
        
        // Update all steps to complete
        STEPS.forEach((step, index) => {
            updateStepProgress(step.id, 'complete');
        });
        
        // Update metrics
        if (data.metrics) {
            updateMetrics(data.metrics);
        }
        
        // Update summary
        if (data.summary) {
            updateSummary(data.summary);
        }
        
        statusText.textContent = '✅ Complete!';
        
    } catch (error) {
        console.error('Pipeline error:', error);
        statusText.textContent = '❌ Error!';
        
        // Mark steps as error
        STEPS.forEach(step => {
            updateStepProgress(step.id, 'error');
        });
        
        alert('Error running pipeline: ' + error.message);
    } finally {
        pipelineRunning = false;
        runButton.disabled = false;
        setTimeout(() => {
            if (statusText.textContent === '✅ Complete!') {
                statusText.textContent = 'Ready to run again';
            }
        }, 2000);
    }
}

// Update step card progress
function updateStepProgress(stepId, status) {
    const card = document.getElementById(stepId);
    if (!card) return;
    
    const statusEl = card.querySelector('.step-status');
    const progressEl = card.querySelector('.progress');
    const detailEl = card.querySelector('.step-detail');
    
    switch(status) {
        case 'pending':
            statusEl.textContent = '⏳';
            statusEl.classList.add('animating');
            if (progressEl) progressEl.style.width = '0%';
            break;
        case 'progress':
            statusEl.textContent = '⏳';
            if (progressEl) progressEl.style.width = '50%';
            break;
        case 'complete':
            statusEl.textContent = '✅';
            statusEl.classList.remove('animating');
            if (progressEl) progressEl.style.width = '100%';
            if (detailEl) detailEl.textContent = 'Completed successfully';
            break;
        case 'error':
            statusEl.textContent = '❌';
            statusEl.classList.remove('animating');
            if (detailEl) detailEl.textContent = 'Failed to complete';
            break;
    }
}

// Reset step card to initial state
function resetStepCard(stepId) {
    const card = document.getElementById(stepId);
    if (!card) return;
    
    const statusEl = card.querySelector('.step-status');
    const progressEl = card.querySelector('.progress');
    const detailEl = card.querySelector('.step-detail');
    const step = STEPS.find(s => s.id === stepId);
    
    statusEl.textContent = '⭕';
    statusEl.classList.remove('animating');
    if (progressEl) progressEl.style.width = '0%';
    if (detailEl && step) detailEl.textContent = step.detail;
}

// Update metrics display
function updateMetrics(metrics) {
    const metricsObj = {
        'metric-sharpe': metrics.sharpe_ratio?.toFixed(4) || 'N/A',
        'metric-sortino': metrics.sortino_ratio?.toFixed(4) || 'N/A',
        'metric-calmar': metrics.calmar_ratio?.toFixed(4) || 'N/A',
        'metric-maxdd': Math.abs(metrics.max_drawdown || 0).toFixed(4) + '%'
    };
    
    Object.entries(metricsObj).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = value;
            el.classList.remove('animating');
        }
    });
}

// Clear metrics display
function clearMetrics() {
    const metricIds = ['metric-sharpe', 'metric-sortino', 'metric-calmar', 'metric-maxdd'];
    metricIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = '...';
            el.classList.add('animating');
        }
    });
}

// Update summary display
function updateSummary(summary) {
    const summaryMap = {
        'summary-stocks': summary.num_stocks || 5,
        'summary-days': summary.num_days || 501,
        'summary-nodes': summary.num_nodes || 5,
        'summary-edges': summary.num_edges || 5,
        'summary-gnn': summary.gnn_hidden_dim || 128,
        'summary-samples': summary.num_samples || 4
    };
    
    Object.entries(summaryMap).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = value;
        }
    });
}

// Sleep helper function
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
