import React, { useMemo } from 'react';
import { analyzeDataset } from '../utils/dataAnalysis';
import { IntelligentAnalyzer } from '../utils/intelligentAnalysis';
import './DatasetOverview.css';

const DatasetOverview = ({ data, fields, fileName }) => {
  const analysis = useMemo(() => {
    return analyzeDataset(data, fields);
  }, [data, fields]);

  const intelligentAnalysis = useMemo(() => {
    if (!data || !fields || data.length === 0) return null;

    const datasetType = IntelligentAnalyzer.detectDatasetType(fields, data);
    return IntelligentAnalyzer.generateBusinessInsights(data, fields, datasetType);
  }, [data, fields]);

  if (!analysis) return null;

  const getQualityColor = (score) => {
    if (score >= 95) return '#10b981'; // Green
    if (score >= 80) return '#f59e0b'; // Yellow
    return '#ef4444'; // Red
  };

  const getQualityLabel = (score) => {
    if (score >= 95) return 'Excellent';
    if (score >= 80) return 'Good';
    return 'Needs Attention';
  };

  return (
    <div className="dataset-overview">
      <div className="overview-header">
        <h3>📋 Dataset Overview</h3>
        {fileName && <span className="file-name">{fileName}</span>}
      </div>

      {/* Intelligent Quick Stats */}
      <div className="quick-stats">
        <div className="stat-item">
          <div className="stat-number">{analysis.rowCount.toLocaleString()}</div>
          <div className="stat-label">
            {intelligentAnalysis?.overview?.primaryLabel?.includes('Shipment') ? 'Shipment Records' :
             intelligentAnalysis?.overview?.primaryLabel?.includes('Sales') ? 'Sales Transactions' :
             intelligentAnalysis?.overview?.primaryLabel?.includes('Performance') ? 'Performance Records' :
             'Rows'}
          </div>
        </div>
        <div className="stat-item">
          <div className="stat-number">{analysis.columnCount}</div>
          <div className="stat-label">
            {intelligentAnalysis?.overview?.secondaryLabel?.includes('Tracking') ? 'Tracking Points' :
             intelligentAnalysis?.overview?.secondaryLabel?.includes('Sales') ? 'Sales Metrics' :
             intelligentAnalysis?.overview?.secondaryLabel?.includes('KPIs') ? 'KPIs' :
             'Columns'}
          </div>
        </div>
        <div className="stat-item">
          <div
            className="stat-number"
            style={{ color: getQualityColor(analysis.dataQuality.score) }}
          >
            {analysis.dataQuality.score}%
          </div>
          <div className="stat-label">
            {intelligentAnalysis?.type === 'shipment' ? 'Tracking Quality' :
             intelligentAnalysis?.type === 'sales' ? 'Data Completeness' :
             'Quality'}
          </div>
        </div>
      </div>

      {/* Data Quality Section */}
      <div className="quality-section">
        <div className="quality-header">
          <span className="quality-icon">🎯</span>
          <span>Data Quality: {getQualityLabel(analysis.dataQuality.score)}</span>
        </div>
        <div className="quality-bar">
          <div
            className="quality-progress"
            style={{
              width: `${analysis.dataQuality.score}%`,
              backgroundColor: getQualityColor(analysis.dataQuality.score)
            }}
          ></div>
        </div>
        {analysis.dataQuality.issues.length > 0 && (
          <div className="quality-issues">
            <span className="issues-title">⚠️ Issues Found:</span>
            {analysis.dataQuality.issues.map((issue, index) => (
              <div key={index} className="issue-item">{issue}</div>
            ))}
          </div>
        )}
      </div>

      {/* Column Analysis */}
      <div className="columns-section">
        <h4>📊 Column Analysis</h4>
        <div className="columns-grid">
          {analysis.columnAnalysis.map((column, index) => (
            <div key={index} className="column-card">
              <div className="column-header">
                <span className="column-name">{column.name}</span>
                <span className={`column-type type-${column.type}`}>
                  {column.type}
                </span>
              </div>
              <div className="column-stats">
                <div className="column-stat">
                  <span className="stat-key">Unique:</span>
                  <span className="stat-value">{column.uniqueCount}</span>
                </div>
                <div className="column-stat">
                  <span className="stat-key">Complete:</span>
                  <span className="stat-value">{column.completeness}%</span>
                </div>
                {column.stats.min !== undefined && (
                  <div className="column-stat">
                    <span className="stat-key">Range:</span>
                    <span className="stat-value">{column.stats.min} - {column.stats.max}</span>
                  </div>
                )}
                {column.stats.mostFrequent && (
                  <div className="column-stat">
                    <span className="stat-key">Top:</span>
                    <span className="stat-value">{column.stats.mostFrequent}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Business Insights */}
      {intelligentAnalysis && (
        <>
          {/* Key Business Metrics */}
          {intelligentAnalysis.keyMetrics && intelligentAnalysis.keyMetrics.length > 0 && (
            <div className="business-metrics-section">
              <h4>📈 Key Business Metrics</h4>
              <div className="business-metrics-grid">
                {intelligentAnalysis.keyMetrics.map((metric, index) => (
                  <div key={index} className="business-metric-card">
                    <div className="metric-icon">{metric.icon}</div>
                    <div className="metric-content">
                      <div className="metric-value">{metric.value}</div>
                      <div className="metric-label">{metric.label}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Intelligent Insights */}
          {(intelligentAnalysis.overview?.description || intelligentAnalysis.trends?.length > 0) && (
            <div className="insights-section">
              <h4>🔍 Intelligent Insights</h4>
              <div className="insights-grid">
                {/* Coverage insights */}
                {intelligentAnalysis.overview?.description && Object.keys(intelligentAnalysis.overview.description).length > 0 && (
                  <div className="insight-card">
                    <h5>📊 Data Coverage</h5>
                    {intelligentAnalysis.overview.description.timeSpan && (
                      <div className="insight-item">
                        <span className="insight-icon">📅</span>
                        <span>Period: {intelligentAnalysis.overview.description.timeSpan}</span>
                      </div>
                    )}
                    {intelligentAnalysis.overview.description.peakActivity && (
                      <div className="insight-item">
                        <span className="insight-icon">🚢</span>
                        <span>Peak: {intelligentAnalysis.overview.description.peakActivity}</span>
                      </div>
                    )}
                    {intelligentAnalysis.overview.description.locationCoverage && (
                      <div className="insight-item">
                        <span className="insight-icon">📍</span>
                        <span>Locations: {intelligentAnalysis.overview.description.locationCoverage}</span>
                      </div>
                    )}
                    {intelligentAnalysis.overview.description.productDiversity && (
                      <div className="insight-item">
                        <span className="insight-icon">📦</span>
                        <span>Products: {intelligentAnalysis.overview.description.productDiversity}</span>
                      </div>
                    )}
                    {intelligentAnalysis.overview.description.topProduct && (
                      <div className="insight-item">
                        <span className="insight-icon">🎯</span>
                        <span>Top Product: {intelligentAnalysis.overview.description.topProduct}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Trend insights */}
                {intelligentAnalysis.trends && intelligentAnalysis.trends.length > 0 && (
                  <div className="insight-card">
                    <h5>📈 Detected Patterns</h5>
                    {intelligentAnalysis.trends.map((trend, index) => (
                      <div key={index} className="insight-item">
                        <span className="insight-icon">{trend.icon}</span>
                        <span>{trend.message}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Anomalies */}
                {intelligentAnalysis.anomalies && intelligentAnalysis.anomalies.length > 0 && (
                  <div className="insight-card">
                    <h5>⚠️ Data Quality Alerts</h5>
                    {intelligentAnalysis.anomalies.slice(0, 3).map((anomaly, index) => (
                      <div key={index} className={`insight-item anomaly-${anomaly.severity}`}>
                        <span className="insight-icon">
                          {anomaly.severity === 'high' ? '🔴' : anomaly.severity === 'medium' ? '🟡' : '🔵'}
                        </span>
                        <span>{anomaly.message}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}

      {/* Smart Chart Suggestions */}
      <div className="suggestions-section">
        <h4>💡 Smart Chart Suggestions</h4>
        <div className="suggestions-grid">
          {/* Use intelligent suggestions if available, fallback to basic analysis */}
          {intelligentAnalysis?.suggestions && intelligentAnalysis.suggestions.length > 0 ? (
            intelligentAnalysis.suggestions.map((suggestion, index) => (
              <div key={index} className={`suggestion-card priority-${suggestion.priority}`}>
                <div className="suggestion-title">{suggestion.title}</div>
                <div className="suggestion-description">{suggestion.description}</div>
                <div className="suggestion-query">
                  💬 Try: "{suggestion.query}"
                </div>
              </div>
            ))
          ) : (
            analysis.suggestions.map((suggestion, index) => (
              <div key={index} className="suggestion-card">
                <div className="suggestion-title">{suggestion.title}</div>
                <div className="suggestion-description">{suggestion.description}</div>
                <div className="suggestion-type">
                  📈 {suggestion.chartType.charAt(0).toUpperCase() + suggestion.chartType.slice(1)} Chart
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Insights */}
      {analysis.insights.length > 0 && (
        <div className="insights-section">
          <h4>🔍 Quick Insights</h4>
          {analysis.insights.map((insight, index) => (
            <div key={index} className="insight-item">{insight}</div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DatasetOverview;