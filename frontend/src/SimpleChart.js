import React, { memo, useState, useCallback } from 'react';
import { GraphicWalker } from '@kanaries/graphic-walker';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    LineElement,
    PointElement,
    Title,
    Tooltip,
    Legend,
    ArcElement
} from 'chart.js';
import { Bar, Line, Pie, Scatter } from 'react-chartjs-2';
import PromptParser from './PromptParser';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    LineElement,
    PointElement,
    ArcElement,
    Title,
    Tooltip,
    Legend
);

// Helper function to generate safe titles without duplicates
const generateSafeTitle = (fieldMapping) => {
    const xField = fieldMapping?.x?.name;
    const yField = fieldMapping?.y?.name;

    if (!xField && !yField) return 'Chart';
    if (!yField || xField === yField) return `${xField || 'Data'} Analysis`;
    return `${xField} vs ${yField}`;
};

const SimpleChart = memo(({ data, fields, enablePromptMode = false }) => {
    const [promptText, setPromptText] = useState('');
    const [promptError, setPromptError] = useState('');
    const [generatedChart, setGeneratedChart] = useState(null);

    const isValidData = data && fields && Array.isArray(data) && Array.isArray(fields) && data.length > 0 && fields.length > 0;

    const handleGenerateChart = useCallback(() => {
        if (!promptText.trim()) {
            setPromptError('Please enter a prompt');
            return;
        }

        try {
            const validation = PromptParser.validatePrompt(promptText, fields);
            if (!validation.valid) {
                setPromptError(validation.error);
                return;
            }

            const parsed = PromptParser.parsePrompt(promptText, fields, data);
            if (parsed.success && parsed.fieldMapping) {
                console.log('Generating Chart.js chart for:', parsed);

                // Create Chart.js data structure
                const chartData = createChartData(parsed, data);
                const chartOptions = createChartOptions(parsed);

                setGeneratedChart({
                    type: parsed.chartType,
                    data: chartData,
                    options: chartOptions,
                    title: generateSafeTitle(parsed.fieldMapping)
                });
                setPromptError('');
            } else {
                setPromptError('Failed to parse prompt');
            }
        } catch (error) {
            console.error('Error:', error);
            setPromptError('Error: ' + error.message);
        }
    }, [promptText, fields, data]);

    // Smart data sampling for performance
    const optimizeDataForChart = (data, chartType, maxPoints = 1000) => {
        if (data.length <= maxPoints) return data;

        console.log(`Performance: Sampling ${maxPoints} points from ${data.length} rows`);

        if (chartType === 'scatter') {
            // For scatter plots, use random sampling to maintain distribution
            const step = Math.floor(data.length / maxPoints);
            return data.filter((_, index) => index % step === 0).slice(0, maxPoints);
        } else {
            // For bar/line/pie charts, we'll aggregate data anyway
            return data;
        }
    };

    // Create Chart.js data structure
    const createChartData = (parsed, rawData) => {
        const { fieldMapping, chartType } = parsed;

        if (!fieldMapping.x) {
            return { labels: [], datasets: [] };
        }

        console.log('Chart type:', chartType);
        console.log('Field mapping:', fieldMapping);
        console.log('Raw data sample:', rawData.slice(0, 5));

        // Optimize data for performance
        const optimizedData = optimizeDataForChart(rawData, chartType);

        let labels = [];
        let values = [];

        if (chartType === 'pie') {
            // For pie charts, use X field for categories and count or sum Y field
            const dataMap = new Map();

            if (fieldMapping.y && fieldMapping.y.analyticType === 'measure') {
                // Sum Y values for each X category
                optimizedData.forEach(row => {
                    const xValue = row[fieldMapping.x.fid] || 'Unknown';
                    const yValue = parseFloat(row[fieldMapping.y.fid]) || 0;

                    if (dataMap.has(xValue)) {
                        dataMap.set(xValue, dataMap.get(xValue) + yValue);
                    } else {
                        dataMap.set(xValue, yValue);
                    }
                });
            } else {
                // Count occurrences of X categories
                optimizedData.forEach(row => {
                    const xValue = row[fieldMapping.x.fid] || 'Unknown';

                    if (dataMap.has(xValue)) {
                        dataMap.set(xValue, dataMap.get(xValue) + 1);
                    } else {
                        dataMap.set(xValue, 1);
                    }
                });
            }

            // Limit pie chart to top 10 categories for performance
            const sortedEntries = Array.from(dataMap.entries())
                .sort(([,a], [,b]) => b - a)
                .slice(0, 10);

            labels = sortedEntries.map(([key]) => key);
            values = sortedEntries.map(([,value]) => value);

            if (dataMap.size > 10) {
                console.log(`Performance: Showing top 10 categories out of ${dataMap.size}`);
            }

        } else {
            // For bar/line charts, need both X and Y
            if (!fieldMapping.y) {
                return { labels: [], datasets: [] };
            }

            const dataMap = new Map();

            if (fieldMapping.x.analyticType === 'dimension' && fieldMapping.y.analyticType === 'measure') {
                // Standard case: categorical X, numerical Y
                optimizedData.forEach(row => {
                    const xValue = row[fieldMapping.x.fid] || 'Unknown';
                    const yValue = parseFloat(row[fieldMapping.y.fid]) || 0;

                    if (dataMap.has(xValue)) {
                        dataMap.set(xValue, dataMap.get(xValue) + yValue);
                    } else {
                        dataMap.set(xValue, yValue);
                    }
                });
            } else if (fieldMapping.x.analyticType === 'measure' && fieldMapping.y.analyticType === 'dimension') {
                // Swap X and Y for scatter plots
                optimizedData.forEach(row => {
                    const xValue = parseFloat(row[fieldMapping.x.fid]) || 0;
                    const yValue = row[fieldMapping.y.fid] || 'Unknown';

                    if (dataMap.has(yValue)) {
                        dataMap.set(yValue, dataMap.get(yValue) + xValue);
                    } else {
                        dataMap.set(yValue, xValue);
                    }
                });
            } else if (fieldMapping.x.analyticType === 'measure' && fieldMapping.y.analyticType === 'measure') {
                // Both numerical - create scatter plot data (already optimized)
                labels = optimizedData.map((_, index) => `Point ${index + 1}`);
                values = optimizedData.map(row => ({
                    x: parseFloat(row[fieldMapping.x.fid]) || 0,
                    y: parseFloat(row[fieldMapping.y.fid]) || 0
                }));
            } else {
                // Both categorical - count combinations
                optimizedData.forEach(row => {
                    const xValue = row[fieldMapping.x.fid] || 'Unknown';
                    if (dataMap.has(xValue)) {
                        dataMap.set(xValue, dataMap.get(xValue) + 1);
                    } else {
                        dataMap.set(xValue, 1);
                    }
                });
            }

            if (chartType !== 'scatter' || fieldMapping.x.analyticType !== 'measure') {
                // Limit bar/line charts to top 20 categories for performance
                if (dataMap.size > 20 && (chartType === 'bar' || chartType === 'line')) {
                    const sortedEntries = Array.from(dataMap.entries())
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 20);

                    labels = sortedEntries.map(([key]) => key);
                    values = sortedEntries.map(([,value]) => value);

                    console.log(`Performance: Showing top 20 categories out of ${dataMap.size}`);
                } else {
                    labels = Array.from(dataMap.keys());
                    values = Array.from(dataMap.values());
                }
            }
        }

        console.log('Generated data:', { labels, values });

        // Professional color schemes
        const colors = {
            bar: 'rgba(59, 130, 246, 0.8)', // Blue
            line: 'rgba(16, 185, 129, 0.8)', // Emerald
            pie: [
                'rgba(59, 130, 246, 0.8)', // Blue
                'rgba(16, 185, 129, 0.8)', // Emerald
                'rgba(245, 158, 11, 0.8)', // Amber
                'rgba(239, 68, 68, 0.8)', // Red
                'rgba(139, 92, 246, 0.8)', // Violet
                'rgba(236, 72, 153, 0.8)', // Pink
                'rgba(6, 182, 212, 0.8)', // Cyan
                'rgba(34, 197, 94, 0.8)' // Green
            ]
        };

        if (chartType === 'pie') {
            return {
                labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors.pie,
                    borderColor: colors.pie.map(color => color.replace('0.8', '1')),
                    borderWidth: 1
                }]
            };
        } else {
            // Handle scatter plots differently
            if (chartType === 'scatter' && values.length > 0 && typeof values[0] === 'object') {
                return {
                    datasets: [{
                        label: generateSafeTitle(fieldMapping),
                        data: values,
                        backgroundColor: colors.line,
                        borderColor: colors.line.replace('0.8', '1'),
                        borderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }]
                };
            } else {
                return {
                    labels,
                    datasets: [{
                        label: fieldMapping.y?.name || fieldMapping.x.name,
                        data: values,
                        backgroundColor: colors[chartType] || colors.bar,
                        borderColor: colors[chartType]?.replace('0.8', '1') || 'rgba(59, 130, 246, 1)',
                        borderWidth: chartType === 'line' ? 3 : 1,
                        fill: chartType === 'area',
                        pointRadius: chartType === 'line' ? 4 : 0,
                        pointHoverRadius: chartType === 'line' ? 6 : 0,
                        tension: chartType === 'line' ? 0.4 : 0
                    }]
                };
            }
        }
    };

    const createChartOptions = (parsed) => {
        const { fieldMapping, chartType } = parsed;

        const baseOptions = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20,
                        font: {
                            size: 12,
                            family: 'Arial, sans-serif'
                        }
                    }
                },
                title: {
                    display: true,
                    text: generateSafeTitle(fieldMapping),
                    font: {
                        size: 16,
                        weight: 'bold',
                        family: 'Arial, sans-serif'
                    },
                    padding: {
                        top: 10,
                        bottom: 30
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    cornerRadius: 6,
                    displayColors: false
                }
            },
            layout: {
                padding: {
                    top: 10,
                    bottom: 10,
                    left: 10,
                    right: 10
                }
            }
        };

        if (chartType === 'pie') {
            return {
                ...baseOptions,
                plugins: {
                    ...baseOptions.plugins,
                    legend: {
                        ...baseOptions.plugins.legend,
                        position: 'right'
                    }
                }
            };
        }

        return {
            ...baseOptions,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: fieldMapping.x?.name || 'X-Axis',
                        font: {
                            size: 14,
                            weight: 'bold',
                            family: 'Arial, sans-serif'
                        }
                    },
                    ticks: {
                        font: {
                            size: 11,
                            family: 'Arial, sans-serif'
                        },
                        maxRotation: 45,
                        minRotation: 0
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: fieldMapping.y?.name || 'Y-Axis',
                        font: {
                            size: 14,
                            weight: 'bold',
                            family: 'Arial, sans-serif'
                        }
                    },
                    ticks: {
                        font: {
                            size: 11,
                            family: 'Arial, sans-serif'
                        },
                        callback: function(value) {
                            if (value >= 1000000) {
                                return (value / 1000000).toFixed(1) + 'M';
                            } else if (value >= 1000) {
                                return (value / 1000).toFixed(1) + 'K';
                            }
                            return value;
                        }
                    },
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        lineWidth: 1
                    }
                }
            }
        };
    };

    // Render Chart.js component based on type
    const renderGeneratedChart = () => {
        if (!generatedChart) return null;

        const props = {
            data: generatedChart.data,
            options: generatedChart.options
        };

        switch (generatedChart.type) {
            case 'line':
                return <Line {...props} />;
            case 'pie':
                return <Pie {...props} />;
            case 'scatter':
                return <Scatter {...props} />;
            case 'bar':
            default:
                return <Bar {...props} />;
        }
    };

    const examplePrompts = PromptParser.generateExamplePrompts(fields);

    if (!isValidData) {
        return <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>No data to display</div>;
    }

    return (
        <div style={{ width: '100%' }}>
            {enablePromptMode && (
                <div style={{
                    marginBottom: '20px',
                    padding: '15px',
                    border: '1px solid #e0e0e0',
                    borderRadius: '8px',
                    backgroundColor: '#f9f9f9'
                }}>
                    <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>
                        🤖 AI-Powered Visualization
                    </h3>

                    <div style={{ marginBottom: '10px' }}>
                        <input
                            type="text"
                            value={promptText}
                            onChange={(e) => setPromptText(e.target.value)}
                            placeholder="e.g., Site ID, Daily Volume, bar chart"
                            style={{
                                width: '70%',
                                padding: '10px',
                                marginRight: '10px',
                                border: '1px solid #ddd',
                                borderRadius: '4px',
                                fontSize: '14px'
                            }}
                            onKeyDown={(e) => e.key === 'Enter' && handleGenerateChart()}
                        />
                        <button
                            onClick={handleGenerateChart}
                            style={{
                                padding: '10px 20px',
                                backgroundColor: '#007bff',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '14px'
                            }}
                        >
                            Generate Chart
                        </button>
                    </div>

                    {promptError && (
                        <div style={{ color: '#d32f2f', fontSize: '14px', marginTop: '10px' }}>
                            {promptError}
                        </div>
                    )}

                    {examplePrompts.length > 0 && (
                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
                            <strong>Examples:</strong>
                            {examplePrompts.map((example, idx) => (
                                <span
                                    key={idx}
                                    onClick={() => setPromptText(example)}
                                    style={{
                                        marginLeft: '10px',
                                        cursor: 'pointer',
                                        color: '#007bff',
                                        textDecoration: 'underline'
                                    }}
                                >
                                    {example}
                                </span>
                            ))}
                        </div>
                    )}

                    {generatedChart && (
                        <div style={{
                            marginTop: '10px',
                            padding: '8px',
                            backgroundColor: '#e8f5e8',
                            borderRadius: '4px',
                            color: '#2e7d32',
                            fontSize: '12px'
                        }}>
                            ✅ Chart generated: "{generatedChart.title}"
                            <button
                                onClick={() => setGeneratedChart(null)}
                                style={{
                                    marginLeft: '10px',
                                    padding: '4px 8px',
                                    fontSize: '11px',
                                    background: 'transparent',
                                    border: '1px solid #2e7d32',
                                    borderRadius: '3px',
                                    cursor: 'pointer',
                                    color: '#2e7d32'
                                }}
                            >
                                Reset
                            </button>
                        </div>
                    )}
                </div>
            )}

            <div style={{
                width: '100%',
                height: '500px',
                border: '1px solid #ccc',
                borderRadius: '8px',
                overflow: 'hidden',
                backgroundColor: '#fff',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}>
                {generatedChart ? (
                    // Show Chart.js chart when prompt is used
                    <div style={{
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        padding: '20px'
                    }}>
                        <div style={{
                            marginBottom: '15px',
                            textAlign: 'center',
                            borderBottom: '2px solid #f0f0f0',
                            paddingBottom: '10px'
                        }}>
                            <h4 style={{
                                margin: '0',
                                color: '#333',
                                fontSize: '14px',
                                fontWeight: 'normal'
                            }}>
                                🎯 Generated: {generatedChart.title}
                            </h4>
                        </div>
                        <div style={{
                            flex: 1,
                            minHeight: '0',
                            position: 'relative',
                            width: '100%',
                            height: 'calc(100% - 50px)'
                        }}>
                            <div style={{
                                position: 'absolute',
                                top: '0',
                                left: '0',
                                right: '0',
                                bottom: '0'
                            }}>
                                {renderGeneratedChart()}
                            </div>
                        </div>
                    </div>
                ) : (
                    // Show GraphicWalker for manual use (with performance optimization)
                    <div>
                        {data.length > 5000 && (
                            <div style={{
                                padding: '10px',
                                backgroundColor: '#fff3cd',
                                border: '1px solid #ffeaa7',
                                borderRadius: '4px',
                                marginBottom: '10px',
                                fontSize: '12px',
                                color: '#856404'
                            }}>
                                ⚡ Performance mode: Showing first 5,000 rows out of {data.length.toLocaleString()} for optimal performance
                            </div>
                        )}
                        <div className="graphic-walker-container">
                            <GraphicWalker
                                data={data.length > 5000 ? data.slice(0, 5000) : data}
                                fields={fields}
                            />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
});

SimpleChart.displayName = 'SimpleChart';
export default SimpleChart;