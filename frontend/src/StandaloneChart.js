import React, { useState, useMemo, useCallback } from 'react';
import * as XLSX from 'xlsx';
import SimpleChart from './SimpleChart';
import { MockTerminalAPI } from './MockAPI';
import DatasetOverview from './components/DatasetOverview';
import ThemeToggle from './components/ThemeToggle';
import AIDashboardGenerator from './components/AIDashboardGenerator';
import useDarkMode from './hooks/useDarkMode';
import PromptParser from './PromptParser';
import { TextWrapper } from './utils/textWrapper';
import './StandaloneChart.css';

// Theme colors (available for future use)
// const theme = {
//     primary: '#1e3a8a',
//     secondary: '#3b82f6',
//     accent: '#10b981',
//     warning: '#f59e0b',
//     danger: '#ef4444',
//     light: '#f8fafc',
//     dark: '#1e293b',
//     white: '#ffffff'
// };

// Memoized sub-components to prevent re-renders
const Header = React.memo(({ isDark, onThemeToggle }) => (
    <header className="app-header">
        <div className="header-container">
            <div className="header-content">
                <div className="logo">TM</div>
                <div className="header-text">
                    <h1>Terminal Manager Analytics</h1>
                    <p>Self-Service Data Visualization Platform powered by GraphicWalker</p>
                </div>
                <div style={{ marginLeft: 'auto' }}>
                    <ThemeToggle isDark={isDark} onToggle={onThemeToggle} />
                </div>
            </div>
        </div>
    </header>
));

const DataSummary = React.memo(({ data, fields }) => {
    if (!data || !fields) return null;
    
    return (
        <div className="data-summary">
            <h4>📈 Data Overview</h4>
            <div className="summary-grid">
                <div className="summary-item">
                    <div className="summary-number">{data.length}</div>
                    <div className="summary-label">Records</div>
                </div>
                <div className="summary-item">
                    <div className="summary-number">{fields.length}</div>
                    <div className="summary-label">Fields</div>
                </div>
            </div>
            <div className="fields-section">
                <p><strong>Available Fields:</strong></p>
                <div className="fields-list">
                    {fields.slice(0, 6).map(field => (
                        <span key={field.name} className="field-tag">
                            {field.name}
                        </span>
                    ))}
                    {fields.length > 6 && (
                        <span className="field-tag-more">
                            +{fields.length - 6} more
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
});

const StandaloneChart = () => {
    const [data, setData] = useState(null);
    const [fields, setFields] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [promptText, setPromptText] = useState('');
    const [fileName, setFileName] = useState('');
    const [chartSpec, setChartSpec] = useState(null);
    const [chartMessage, setChartMessage] = useState('');
    const { isDark, toggleTheme } = useDarkMode();

    // Memoized data cleaning function
    const cleanData = useCallback((rawData) => {
        if (!rawData || !Array.isArray(rawData)) return [];

        const cleaned = rawData.map(row => {
            if (!row || typeof row !== 'object') return null;

            const cleanRow = {};
            Object.keys(row).forEach(key => {
                if (!key || key.trim() === '') return;

                let value = row[key];
                if (value === null || value === undefined || value === 'null' || value === 'undefined') {
                    cleanRow[key] = '';
                } else if (typeof value === 'number' && !isNaN(value)) {
                    cleanRow[key] = value;
                } else {
                    cleanRow[key] = String(value).trim();
                }
            });
            return cleanRow;
        }).filter(row => row !== null && Object.keys(row).length > 0);

        console.log(`Data cleaning: ${rawData.length} rows -> ${cleaned.length} rows`);
        return cleaned;
    }, []);

    // Memoized API data handlers
    const loadTerminalSites = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await MockTerminalAPI.getTerminalSites();
            if (response.success && response.data) {
                const terminalFields = [
                    { fid: 'siteId', name: 'Site ID', semanticType: 'nominal', analyticType: 'dimension' },
                    { fid: 'siteName', name: 'Site Name', semanticType: 'nominal', analyticType: 'dimension' },
                    { fid: 'fuelType', name: 'Fuel Type', semanticType: 'nominal', analyticType: 'dimension' },
                    { fid: 'dailyVolume', name: 'Daily Volume', semanticType: 'quantitative', analyticType: 'measure' },
                    { fid: 'region', name: 'Region', semanticType: 'nominal', analyticType: 'dimension' },
                    { fid: 'operatingHours', name: 'Operating Hours', semanticType: 'quantitative', analyticType: 'measure' },
                    { fid: 'status', name: 'Status', semanticType: 'nominal', analyticType: 'dimension' }
                ];

                const cleanedData = cleanData(response.data);
                setData(cleanedData);
                setFields(terminalFields);
                setFileName('Terminal Manager Sites (REST API)');
            }
        } catch (error) {
            console.error('Error loading terminal data:', error);
            alert('Error loading terminal data from REST API');
        } finally {
            setIsLoading(false);
        }
    }, [cleanData]);

    const loadSalesAnalytics = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await MockTerminalAPI.getSalesAnalytics();
            if (response.success && response.data) {
                const salesFields = [
                    { fid: 'month', name: 'Month', semanticType: 'nominal', analyticType: 'dimension' },
                    { fid: 'totalSales', name: 'Total Sales', semanticType: 'quantitative', analyticType: 'measure' },
                    { fid: 'region', name: 'Region', semanticType: 'nominal', analyticType: 'dimension' },
                    { fid: 'fuelType', name: 'Fuel Type', semanticType: 'nominal', analyticType: 'dimension' },
                    { fid: 'profit', name: 'Profit', semanticType: 'quantitative', analyticType: 'measure' }
                ];
                const cleanedData = cleanData(response.data);
                setData(cleanedData);
                setFields(salesFields);
                setFileName('Sales Analytics (REST API)');
            }
        } catch (error) {
            console.error('Error loading sales data:', error);
            alert('Error loading sales analytics');
        } finally {
            setIsLoading(false);
        }
    }, [cleanData]);

    const loadPerformanceMetrics = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await MockTerminalAPI.getPerformanceMetrics();
            if (response.success && response.data) {
                const performanceFields = [
                    { fid: 'terminal', name: 'Terminal', semanticType: 'nominal', analyticType: 'dimension' },
                    { fid: 'efficiency', name: 'Efficiency %', semanticType: 'quantitative', analyticType: 'measure' },
                    { fid: 'uptime', name: 'Uptime %', semanticType: 'quantitative', analyticType: 'measure' },
                    { fid: 'throughput', name: 'Throughput', semanticType: 'quantitative', analyticType: 'measure' },
                    { fid: 'quality', name: 'Quality %', semanticType: 'quantitative', analyticType: 'measure' }
                ];
                const cleanedData = cleanData(response.data);
                setData(cleanedData);
                setFields(performanceFields);
                setFileName('Performance Metrics (REST API)');
            }
        } catch (error) {
            console.error('Error loading performance data:', error);
            alert('Error loading performance metrics');
        } finally {
            setIsLoading(false);
        }
    }, [cleanData]);

    // Memoized file upload handler
    const handleFileUpload = useCallback((event) => {
        const file = event.target.files[0];
        if (!file) return;

        const fileType = file.name.split('.').pop().toLowerCase();
        if (!['xlsx', 'xls'].includes(fileType)) {
            alert('Please upload only Excel files (.xlsx or .xls)');
            event.target.value = '';
            return;
        }

        setIsLoading(true);
        setFileName(file.name);

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = new Uint8Array(e.target.result);
                const workbook = XLSX.read(data, { type: 'array' });
                const firstSheet = workbook.Sheets[workbook.SheetNames[0]];
                const excelData = XLSX.utils.sheet_to_json(firstSheet, {
                    raw: false,
                    defval: '',
                    blankrows: false
                });

                if (excelData.length > 0) {
                    console.log('Raw Excel Data:', excelData.slice(0, 3)); // Show first 3 rows
                    console.log('Total rows:', excelData.length);

                    const newFields = Object.keys(excelData[0]).map(key => {
                        const sampleValue = excelData[0][key];
                        const isNumeric = typeof sampleValue === 'number' ||
                                        (!isNaN(Number(sampleValue)) && sampleValue !== '' && sampleValue !== null);
                        console.log(`Field: ${key} | Sample: ${sampleValue} | Type: ${isNumeric ? 'measure' : 'dimension'}`);
                        return {
                            fid: String(key),
                            name: String(key),
                            semanticType: isNumeric ? 'quantitative' : 'nominal',
                            analyticType: isNumeric ? 'measure' : 'dimension',
                        };
                    });

                    const cleanedData = cleanData(excelData);
                    console.log('Cleaned Data:', cleanedData.slice(0, 3)); // Show first 3 rows
                    console.log('Fields:', newFields);

                    setData(cleanedData);
                    setFields(newFields);
                }
            } catch (error) {
                console.error('Error processing Excel file:', error);
                alert('Failed to process Excel file');
            } finally {
                setIsLoading(false);
            }
        };

        reader.onerror = () => {
            alert('Failed to read file');
            setIsLoading(false);
        };

        reader.readAsArrayBuffer(file);
        event.target.value = '';
    }, [cleanData]);

    // Enhanced prompt handler with simple English to technical query conversion
    const handlePromptSubmit = useCallback(() => {
        if (!promptText.trim()) {
            alert('Please enter what you want to visualize');
            return;
        }

        if (!data || !fields) {
            alert('Please load data first');
            return;
        }

        console.log('Original input:', promptText);

        // First, convert simple English to technical query using TextWrapper
        const wrapperResult = TextWrapper.convertToTechnicalQuery(promptText, fields);

        if (!wrapperResult.success) {
            setChartMessage(`❌ ${wrapperResult.error}\n\n💡 Try phrases like:\n• Compare quantity across regions\n• Show sales trends over time\n• Distribution of fuel types`);
            return;
        }

        console.log('Converted to technical query:', wrapperResult.technicalQuery);

        // Then use the existing PromptParser with the technical query
        const result = PromptParser.parsePrompt(wrapperResult.technicalQuery, fields, data);

        if (result.success) {
            // Update chart with parsed specification
            setChartSpec(result.chartSpec);
            setChartMessage(`✅ Chart generated successfully\n📊 Query: ${wrapperResult.technicalQuery}`);
        } else {
            // Show error with suggestions
            let message = `❌ Could not process: "${promptText}"`;
            if (result.suggestions && result.suggestions.length > 0) {
                message += `\n\n💡 Try: ${result.suggestions.join(', ')}`;
            }
            setChartMessage(message);
        }

        setPromptText('');
    }, [promptText, data, fields]);

    // Memoized components to prevent unnecessary re-renders
    const controlPanel = useMemo(() => (
        <div className="control-panel">
            <h3>📊 Data Sources & Configuration</h3>

            {/* REST API Data Sources */}
            <div className="api-section">
                <h4>Terminal Manager REST APIs</h4>
                <div className="api-buttons">
                    <button onClick={loadTerminalSites} disabled={isLoading} className="btn-terminal">
                        🏢 Terminal Sites Data
                    </button>
                    <button onClick={loadSalesAnalytics} disabled={isLoading} className="btn-sales">
                        💰 Sales Analytics
                    </button>
                    <button onClick={loadPerformanceMetrics} disabled={isLoading} className="btn-performance">
                        📊 Performance Metrics
                    </button>
                </div>
                <p className="api-description">Live simulation of Terminal Manager REST API endpoints</p>
            </div>

            {/* File Upload */}
            <div className="upload-section">
                <h4>Upload Site Data</h4>
                <div className="upload-area">
                    <input
                        type="file"
                        accept=".xlsx,.xls"
                        onChange={handleFileUpload}
                        style={{ display: 'none' }}
                        id="fileInput"
                        disabled={isLoading}
                    />
                    <label htmlFor="fileInput" className="upload-label">
                        📁 Click to Upload Excel File
                    </label>
                    <p className="upload-description">Upload your terminal site data (.xlsx, .xls)</p>
                </div>
                {fileName && (
                    <div className="file-loaded">
                        ✓ Loaded: {fileName}
                    </div>
                )}
            </div>

            {/* AI Prompt Section */}
            {(data && fields) && (
                <div className="prompt-section">
                    <h4>🤖 AI-Powered Visualization</h4>
                    <div className="prompt-examples">
                        <p><strong>Examples:</strong></p>
                        <div className="examples-text">
                            <p>• Compare gross quantity across different flow rates</p>
                            <p>• Show sales trends over time by region</p>
                            <p>• Distribution of fuel types by volume</p>
                        </div>
                    </div>
                    <input
                        type="text"
                        placeholder="Describe what you want to visualize..."
                        value={promptText}
                        onChange={(e) => setPromptText(e.target.value)}
                        className="prompt-input"
                    />
                    <button onClick={handlePromptSubmit} className="btn-generate">
                        ✨ Generate Insights
                    </button>
                </div>
            )}

            <DataSummary data={data} fields={fields} />
        </div>
    ), [data, fields, fileName, promptText, isLoading, loadTerminalSites, loadSalesAnalytics, loadPerformanceMetrics, handleFileUpload, handlePromptSubmit]);

    return (
        <div className="app-container">
            <Header isDark={isDark} onThemeToggle={toggleTheme} />
            <div className="main-content">
                {/* Dataset Overview - Show when data is loaded */}
                {(data && fields) && (
                    <DatasetOverview data={data} fields={fields} fileName={fileName} />
                )}

                <div className="dashboard-grid">
                    {controlPanel}

                    {/* Visualization Area */}
                    <div className="visualization-area">
                        <div className="viz-header">
                            <h3>📊 Interactive Analytics Dashboard</h3>
                            <div className="viz-badges">
                                <span className="badge badge-accent">GraphicWalker Enabled</span>
                                <span className="badge badge-secondary">Drag & Drop Ready</span>
                            </div>
                        </div>

                        {isLoading ? (
                            <div className="loading-container">
                                <div className="spinner"></div>
                                <p>Processing your data...</p>
                            </div>
                        ) : (data && fields) ? (
                            <div>
                                <div className="demo-outcome">
                                    <p><strong>🎯 Demonstration Outcome Achieved:</strong></p>
                                    <ul>
                                        <li>✅ <strong>Customized Widgets Support:</strong> GraphicWalker provides drag-and-drop widget creation</li>
                                        <li>✅ <strong>Data Analysis Capabilities:</strong> Interactive visualization with imported data</li>
                                        <li>✅ <strong>Performance Handling:</strong> Efficient rendering of terminal data without timeouts</li>
                                    </ul>
                                    {chartMessage && (
                                        <div className="chart-message">
                                            {chartMessage}
                                        </div>
                                    )}
                                </div>
                                <SimpleChart
                                    data={data}
                                    fields={fields}
                                    chartSpec={chartSpec}
                                    enablePromptMode={true}
                                />
                            </div>
                        ) : (
                            <div className="empty-state">
                                <div className="empty-icon">📊</div>
                                <h4>Ready for Terminal Data Analysis</h4>
                                <p>
                                    Load terminal data or upload site-specific files to start creating
                                    interactive dashboards with drag-and-drop analytics.
                                </p>
                                <div className="features">
                                    <div className="feature">
                                        <div className="feature-icon">🔗</div>
                                        <p>REST API<br/>Integration</p>
                                    </div>
                                    <div className="feature">
                                        <div className="feature-icon">🖱️</div>
                                        <p>Drag & Drop<br/>Analytics</p>
                                    </div>
                                    <div className="feature">
                                        <div className="feature-icon">⚡</div>
                                        <p>Real-time<br/>Performance</p>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* AI Dashboard Generator - New Feature */}
                {(data && fields) && (
                    <AIDashboardGenerator
                        data={data}
                        fileName={fileName}
                    />
                )}

            </div>
        </div>
    );
};

export default StandaloneChart;
