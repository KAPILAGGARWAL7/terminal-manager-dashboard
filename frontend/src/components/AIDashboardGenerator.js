import React, { useState } from 'react';

const AIDashboardGenerator = ({ data, fileName }) => {
    const [prompt, setPrompt] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);
    const [dashboardUrl, setDashboardUrl] = useState('');
    const [dashboardId, setDashboardId] = useState('');
    const [error, setError] = useState('');
    const [excelPath, setExcelPath] = useState('');

    const handleGenerateDashboard = async () => {
        if (!prompt.trim()) {
            setError('Please enter a prompt for your dashboard');
            return;
        }

        setIsGenerating(true);
        setError('');

        try {
            let uploadedExcelPath = '';
            // TODO: make this upload process cleaner
            if (data && fileName) {
                console.log('Uploading Excel data to simple backend...');
                const XLSX = require('xlsx');  // using xlsx library
                const workbook = XLSX.utils.book_new();
                const worksheet = XLSX.utils.json_to_sheet(data);
                XLSX.utils.book_append_sheet(workbook, worksheet, 'Sheet1');
                const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
                const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
                const formData = new FormData();
                formData.append('file', blob, fileName || 'dashboard_data.xlsx');
                formData.append('datasetName', `dashboard_data_${Date.now()}`);
                const uploadResponse = await fetch('http://localhost:5246/Dataset/Upload', {
                    method: 'POST',
                    body: formData
                });
                if (uploadResponse.ok) {
                    const uploadResult = await uploadResponse.json();
                    uploadedExcelPath = uploadResult.filePath;
                    console.log('Excel data uploaded to:', uploadedExcelPath);
                } else {
                    console.warn('Failed to upload Excel data to simple backend, proceeding with data in request');
                }
            }

            // quick timeout setup for the AI call
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 120000);  // 2 min timeout
            if (dashboardId.length > 0)
                await fetch(`http://localhost:5247/api/dashboard/stop/${dashboardId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    signal: controller.signal
                });
                setDashboardId('');
                setDashboardUrl('');
            const response = await fetch('http://localhost:5247/api/dashboard/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    excel_path: uploadedExcelPath,  // file path or empty
                    data: uploadedExcelPath ? undefined : data,  // fallback data
                    fileName: fileName
                }),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            const result = await response.json();

            if (result.success) {
                console.log('Dashboard generated, embed URL:', result.embed_url);
                setDashboardId(result.dashboard_id);
                setDashboardUrl(result.embed_url);
                setError('');
                if (result.ai_generated === false) {
                    setError(`Info: ${result.message}`);
                }
            } else {
                setError(result.error || 'Failed to generate dashboard');
            }
        } catch (err) {
            // basic error handling
            if (err.name === 'AbortError') {
                setError('Request timeout: Dashboard generation is taking longer than expected. The AI backend may be processing - please try again.');
            } else if (err.message.includes('fetch')) {
                setError('Error connecting to AI backend. Make sure Python backend is running.');  // common issue
            } else {
                setError(`Dashboard generation failed: ${err.message}`);
            }
        } finally {
            setIsGenerating(false);  // reset loading state
        }
    };

    const handleCloseDashboard = () => {
        setDashboardUrl('');
        setPrompt('');
    };

    return (
        <div className="ai-dashboard-section">
            <div className="ai-header">
                <h4>AI Dashboard Generator</h4>
                <p>Describe what kind of dashboard you want, and AI will create it for you!</p>
                {/* TODO: add more detailed instructions */}
            </div>

            <div className="ai-controls">
                <div className="prompt-section">
                    <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Example: Create a comprehensive sales dashboard with regional breakdowns, fuel type analysis, and performance metrics with interactive filters..."
                        className="ai-prompt-input"
                        rows={3}
                        disabled={isGenerating}
                    />
                </div>

                <div className="ai-actions">
                    <button
                        onClick={handleGenerateDashboard}
                        disabled={isGenerating || !data}
                        className="btn-ai-generate"
                    >
                        {isGenerating ? (
                            <>
                                <span className="spinner-mini"></span>
                                Generating Dashboard... (This may take up to 2 minutes)
                            </>
                        ) : (
                            'Generate AI Dashboard'
                        )}
                    </button>
                </div>

                {error && (
                    <div className={error.startsWith('Info:') ? "ai-info" : "ai-error"}>
                        {error.startsWith('Info:') ? '' : 'Warning: '}{error}
                    </div>
                )}

                <div className="ai-examples">
                    <p><strong>Try these examples:</strong></p>
                    {/* quick preset buttons for demo */}
                    <div className="example-prompts">
                        <button
                            className="example-prompt"
                            onClick={() => setPrompt("Create a sales performance dashboard with regional comparisons and trend analysis")}
                        >
                            Sales Performance Dashboard
                        </button>
                        <button
                            className="example-prompt"
                            onClick={() => setPrompt("Build an operational efficiency dashboard with KPIs, uptime metrics, and fuel volume analysis")}
                        >
                            Operational Efficiency Dashboard
                        </button>
                        <button
                            className="example-prompt"
                            onClick={() => setPrompt("Generate a financial overview with profit margins, cost analysis, and revenue trends")}
                        >
                            Financial Overview Dashboard
                        </button>
                        <button
                            className="example-prompt"
                            onClick={() => setPrompt("Create a logistics dashboard with shipment tracking, delivery performance, and route optimization")}
                        >
                            Logistics & Supply Chain Dashboard
                        </button>
                        <button
                            className="example-prompt"
                            onClick={() => setPrompt("Build an analytics dashboard with data exploration, trends, and correlation analysis")}
                        >
                            Data Analytics Dashboard
                        </button>
                        <button
                            className="example-prompt"
                            onClick={() => setPrompt("Generate an energy consumption dashboard with efficiency metrics and cost analysis")}
                        >
                            Energy Consumption Dashboard
                        </button>
                        <button
                            className="example-prompt"
                            onClick={() => setPrompt("Create an HR dashboard with employee metrics, performance analysis, and workforce demographics")}
                        >
                            HR Analytics Dashboard
                        </button>
                    </div>
                </div>
            </div>

            {dashboardUrl && (
                <div className="ai-dashboard-container">
                    <div className="dashboard-header">
                        <h5>Generated Dashboard</h5>
                        <button onClick={handleCloseDashboard} className="close-dashboard">
                            Close
                        </button>
                    </div>
                    <div className="dashboard-iframe-container">
                        <iframe
                            src={dashboardUrl}
                            width="100%"
                            height="600"
                            frameBorder="0"
                            style={{
                                borderRadius: '8px',
                                boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
                            }}
                        />
                    </div>
                    <div className="dashboard-footer">
                        <a
                            href={dashboardUrl.replace('?embed=true', '')}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="open-fullscreen"
                        >
                            Open in Full Screen
                        </a>
                    </div>
                </div>
            )}

            <style jsx>{`
                .ai-dashboard-section {
                    margin-top: 2rem;
                    padding: 1.5rem;
                    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                    border-radius: 12px;
                    border: 1px solid #cbd5e1;
                }

                .ai-header h4 {
                    margin: 0 0 0.5rem 0;
                    color: #1e293b;
                    font-size: 1.2rem;
                    font-weight: 600;
                }

                .ai-header p {
                    margin: 0 0 1rem 0;
                    color: #64748b;
                    font-size: 0.9rem;
                }

                .ai-prompt-input {
                    width: 100%;
                    padding: 1rem;
                    border: 2px solid #e2e8f0;
                    border-radius: 8px;
                    font-size: 0.9rem;
                    font-family: inherit;
                    resize: vertical;
                    min-height: 80px;
                    background: white;
                    transition: border-color 0.2s;
                }

                .ai-prompt-input:focus {
                    outline: none;
                    border-color: #3b82f6;
                    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
                }

                .ai-actions {
                    margin: 1rem 0;
                }

                .btn-ai-generate {
                    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                    color: white;
                    border: none;
                    padding: 0.75rem 1.5rem;
                    border-radius: 8px;
                    font-size: 0.9rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                .btn-ai-generate:hover:not(:disabled) {
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
                }

                .btn-ai-generate:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                }

                .spinner-mini {
                    width: 12px;
                    height: 12px;
                    border: 2px solid transparent;
                    border-top: 2px solid white;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }

                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }

                .ai-error {
                    background: #fef2f2;
                    border: 1px solid #fecaca;
                    color: #dc2626;
                    padding: 0.75rem;
                    border-radius: 6px;
                    margin: 0.5rem 0;
                    font-size: 0.9rem;
                }

                .ai-info {
                    background: #eff6ff;
                    border: 1px solid #bfdbfe;
                    color: #1d4ed8;
                    padding: 0.75rem;
                    border-radius: 6px;
                    margin: 0.5rem 0;
                    font-size: 0.9rem;
                }

                .ai-examples {
                    margin-top: 1rem;
                    padding-top: 1rem;
                    border-top: 1px solid #e2e8f0;
                }

                .ai-examples p {
                    margin: 0 0 0.5rem 0;
                    color: #64748b;
                    font-size: 0.85rem;
                    font-weight: 500;
                }

                .example-prompts {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.5rem;
                }

                .example-prompt {
                    background: white;
                    border: 1px solid #d1d5db;
                    color: #374151;
                    padding: 0.5rem 0.75rem;
                    border-radius: 6px;
                    font-size: 0.8rem;
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .example-prompt:hover {
                    border-color: #3b82f6;
                    background: #eff6ff;
                    color: #1d4ed8;
                }

                .ai-dashboard-container {
                    margin-top: 1.5rem;
                    background: white;
                    border-radius: 12px;
                    padding: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                }

                .dashboard-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1rem;
                    padding-bottom: 0.5rem;
                    border-bottom: 1px solid #e2e8f0;
                }

                .dashboard-header h5 {
                    margin: 0;
                    color: #1e293b;
                    font-size: 1rem;
                    font-weight: 600;
                }

                .close-dashboard {
                    background: #f1f5f9;
                    border: 1px solid #cbd5e1;
                    color: #64748b;
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    font-size: 0.8rem;
                    cursor: pointer;
                    transition: all 0.2s;
                }

                .close-dashboard:hover {
                    background: #e2e8f0;
                    color: #374151;
                }

                .dashboard-iframe-container {
                    position: relative;
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                }

                .dashboard-footer {
                    margin-top: 0.75rem;
                    text-align: center;
                }

                .open-fullscreen {
                    color: #3b82f6;
                    text-decoration: none;
                    font-size: 0.9rem;
                    font-weight: 500;
                    transition: color 0.2s;
                }

                .open-fullscreen:hover {
                    color: #1d4ed8;
                    text-decoration: underline;
                }
            `}</style>
        </div>
    );
};

export default AIDashboardGenerator;