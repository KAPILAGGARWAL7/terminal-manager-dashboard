/**
 * Intelligent Dataset Analysis Engine
 * Provides business-context aware analysis for shipment, sales, and operational data
 */

export class IntelligentAnalyzer {

  /**
   * Detect the type of dataset based on field names and data patterns
   */
  static detectDatasetType(fields, data) {
    const fieldNames = fields.map(f => f.name.toLowerCase());

    // Shipment data indicators
    const shipmentIndicators = [
      'shipment', 'compartment', 'flowrate', 'grossquantity',
      'exittime', 'scheduleddate', 'baycode', 'baseproduct'
    ];

    // Sales data indicators
    const salesIndicators = ['sales', 'revenue', 'price', 'customer', 'order'];

    // Performance data indicators
    const performanceIndicators = ['efficiency', 'uptime', 'throughput', 'performance'];

    const shipmentScore = this.calculateMatchScore(fieldNames, shipmentIndicators);
    const salesScore = this.calculateMatchScore(fieldNames, salesIndicators);
    const performanceScore = this.calculateMatchScore(fieldNames, performanceIndicators);

    if (shipmentScore > salesScore && shipmentScore > performanceScore) {
      return 'shipment';
    } else if (salesScore > performanceScore) {
      return 'sales';
    } else if (performanceScore > 0) {
      return 'performance';
    }

    return 'general';
  }

  static calculateMatchScore(fieldNames, indicators) {
    return indicators.reduce((score, indicator) => {
      return score + fieldNames.filter(name => name.includes(indicator)).length;
    }, 0);
  }

  /**
   * Generate intelligent business insights based on data type and content
   */
  static generateBusinessInsights(data, fields, datasetType) {
    const insights = {
      type: datasetType,
      overview: this.generateOverview(data, fields, datasetType),
      keyMetrics: this.generateKeyMetrics(data, fields, datasetType),
      trends: this.detectTrends(data, fields, datasetType),
      anomalies: this.detectAnomalies(data, fields),
      suggestions: this.generateSuggestions(data, fields, datasetType)
    };

    return insights;
  }

  /**
   * Generate contextual overview based on dataset type
   */
  static generateOverview(data, fields, datasetType) {
    const rowCount = data.length;
    const columnCount = fields.length;

    const overviews = {
      shipment: {
        primaryLabel: `${rowCount.toLocaleString()} Shipment Records`,
        secondaryLabel: `${columnCount} Tracking Points`,
        description: this.analyzeShipmentCoverage(data, fields)
      },
      sales: {
        primaryLabel: `${rowCount.toLocaleString()} Sales Transactions`,
        secondaryLabel: `${columnCount} Sales Metrics`,
        description: this.analyzeSalesCoverage(data, fields)
      },
      performance: {
        primaryLabel: `${rowCount.toLocaleString()} Performance Records`,
        secondaryLabel: `${columnCount} KPIs`,
        description: this.analyzePerformanceCoverage(data, fields)
      },
      general: {
        primaryLabel: `${rowCount.toLocaleString()} Data Records`,
        secondaryLabel: `${columnCount} Variables`,
        description: 'General dataset analysis'
      }
    };

    return overviews[datasetType] || overviews.general;
  }

  /**
   * Analyze shipment data coverage and patterns
   */
  static analyzeShipmentCoverage(data, fields) {
    const analysis = {};

    // Find date-related fields
    const dateFields = fields.filter(f =>
      f.name.toLowerCase().includes('time') ||
      f.name.toLowerCase().includes('date') ||
      f.semanticType === 'temporal'
    );

    // Find location/terminal fields
    const locationFields = fields.filter(f =>
      f.name.toLowerCase().includes('terminal') ||
      f.name.toLowerCase().includes('bay') ||
      f.name.toLowerCase().includes('compartment')
    );

    // Find product fields
    const productFields = fields.filter(f =>
      f.name.toLowerCase().includes('product') ||
      f.name.toLowerCase().includes('base')
    );

    // Analyze temporal coverage
    if (dateFields.length > 0) {
      const dateField = dateFields[0];
      const dates = data.map(row => new Date(row[dateField.name])).filter(d => !isNaN(d));
      if (dates.length > 0) {
        const minDate = new Date(Math.min(...dates));
        const maxDate = new Date(Math.max(...dates));
        analysis.timeSpan = `${minDate.toLocaleDateString()} - ${maxDate.toLocaleDateString()}`;

        // Detect peak activity
        const monthCounts = this.groupByMonth(dates);
        const peakMonth = Object.entries(monthCounts)
          .sort(([,a], [,b]) => b - a)[0];
        if (peakMonth) {
          analysis.peakActivity = `${peakMonth[0]} (${peakMonth[1].toLocaleString()} shipments)`;
        }
      }
    }

    // Analyze geographic coverage
    if (locationFields.length > 0) {
      const locationField = locationFields[0];
      const uniqueLocations = [...new Set(data.map(row => row[locationField.name]))].filter(Boolean);
      analysis.locationCoverage = `${uniqueLocations.length} unique locations`;
    }

    // Analyze product diversity
    if (productFields.length > 0) {
      const productField = productFields[0];
      const uniqueProducts = [...new Set(data.map(row => row[productField.name]))].filter(Boolean);
      analysis.productDiversity = `${uniqueProducts.length} product types`;

      // Find most common product
      const productCounts = this.countOccurrences(data, productField.name);
      const topProduct = Object.entries(productCounts)
        .sort(([,a], [,b]) => b - a)[0];
      if (topProduct) {
        const percentage = ((topProduct[1] / data.length) * 100).toFixed(0);
        analysis.topProduct = `${topProduct[0]} (${percentage}% of volume)`;
      }
    }

    return analysis;
  }

  /**
   * Generate key metrics based on dataset type
   */
  static generateKeyMetrics(data, fields, datasetType) {
    const metrics = [];

    if (datasetType === 'shipment') {
      // Shipment-specific metrics
      const quantityField = fields.find(f =>
        f.name.toLowerCase().includes('quantity') ||
        f.name.toLowerCase().includes('volume')
      );

      if (quantityField) {
        const quantities = data.map(row => parseFloat(row[quantityField.name])).filter(q => !isNaN(q));
        if (quantities.length > 0) {
          const totalVolume = quantities.reduce((sum, q) => sum + q, 0);
          const avgVolume = totalVolume / quantities.length;

          metrics.push({
            label: 'Total Volume',
            value: totalVolume.toLocaleString(undefined, { maximumFractionDigits: 0 }),
            icon: '📦'
          });

          metrics.push({
            label: 'Avg per Shipment',
            value: avgVolume.toLocaleString(undefined, { maximumFractionDigits: 1 }),
            icon: '📊'
          });
        }
      }

      // Flow rate analysis
      const flowField = fields.find(f => f.name.toLowerCase().includes('flow'));
      if (flowField) {
        const flowRates = data.map(row => parseFloat(row[flowField.name])).filter(f => !isNaN(f));
        if (flowRates.length > 0) {
          const avgFlow = flowRates.reduce((sum, f) => sum + f, 0) / flowRates.length;
          metrics.push({
            label: 'Avg Flow Rate',
            value: avgFlow.toLocaleString(undefined, { maximumFractionDigits: 2 }),
            icon: '⚡'
          });
        }
      }
    }

    return metrics;
  }

  /**
   * Detect trends in the data
   */
  static detectTrends(data, fields, datasetType) {
    const trends = [];

    // Time-based trend analysis
    const timeField = fields.find(f =>
      f.name.toLowerCase().includes('time') ||
      f.name.toLowerCase().includes('date') ||
      f.semanticType === 'temporal'
    );

    if (timeField && datasetType === 'shipment') {
      const timeAnalysis = this.analyzeTimePatterns(data, timeField);
      trends.push(...timeAnalysis);
    }

    return trends;
  }

  /**
   * Detect anomalies in the data
   */
  static detectAnomalies(data, fields) {
    const anomalies = [];

    // Check for missing data
    fields.forEach(field => {
      const nullCount = data.filter(row =>
        !row[field.name] ||
        row[field.name] === '' ||
        row[field.name] === null ||
        row[field.name] === undefined
      ).length;

      if (nullCount > 0) {
        const percentage = ((nullCount / data.length) * 100).toFixed(1);
        if (percentage > 5) { // Only report significant missing data
          anomalies.push({
            type: 'missing_data',
            message: `${percentage}% missing values in ${field.name}`,
            severity: percentage > 20 ? 'high' : 'medium'
          });
        }
      }
    });

    // Detect outliers in numeric fields
    const numericFields = fields.filter(f => f.analyticType === 'measure');
    numericFields.forEach(field => {
      const values = data.map(row => parseFloat(row[field.name])).filter(v => !isNaN(v));
      if (values.length > 10) {
        const outliers = this.detectOutliers(values);
        if (outliers.length > 0) {
          anomalies.push({
            type: 'outliers',
            message: `${outliers.length} potential outliers detected in ${field.name}`,
            severity: 'low'
          });
        }
      }
    });

    return anomalies;
  }

  /**
   * Generate smart suggestions for charts and analysis
   */
  static generateSuggestions(data, fields, datasetType) {
    const suggestions = [];

    if (datasetType === 'shipment') {
      // Time-based suggestions
      const timeField = fields.find(f =>
        f.name.toLowerCase().includes('time') ||
        f.name.toLowerCase().includes('date')
      );

      const quantityField = fields.find(f =>
        f.name.toLowerCase().includes('quantity') ||
        f.name.toLowerCase().includes('volume')
      );

      if (timeField && quantityField) {
        suggestions.push({
          type: 'chart',
          title: 'Volume Trends Over Time',
          description: `Track ${quantityField.name} patterns across ${timeField.name}`,
          query: `${quantityField.name}, ${timeField.name}, line chart`,
          priority: 'high'
        });
      }

      // Location-based suggestions
      const locationField = fields.find(f =>
        f.name.toLowerCase().includes('terminal') ||
        f.name.toLowerCase().includes('bay') ||
        f.name.toLowerCase().includes('compartment')
      );

      if (locationField && quantityField) {
        suggestions.push({
          type: 'chart',
          title: 'Volume by Location',
          description: `Compare ${quantityField.name} across ${locationField.name}`,
          query: `${locationField.name}, ${quantityField.name}, bar chart`,
          priority: 'high'
        });
      }

      // Product analysis
      const productField = fields.find(f =>
        f.name.toLowerCase().includes('product')
      );

      if (productField) {
        suggestions.push({
          type: 'chart',
          title: 'Product Distribution',
          description: `Analyze distribution of ${productField.name}`,
          query: `${productField.name}, pie chart`,
          priority: 'medium'
        });
      }

      // Flow rate correlation
      const flowField = fields.find(f => f.name.toLowerCase().includes('flow'));
      if (flowField && quantityField) {
        suggestions.push({
          type: 'chart',
          title: 'Flow Rate vs Volume',
          description: `Explore relationship between ${flowField.name} and ${quantityField.name}`,
          query: `${flowField.name}, ${quantityField.name}, scatter plot`,
          priority: 'medium'
        });
      }
    }

    return suggestions.slice(0, 4); // Limit to top 4 suggestions
  }

  // Helper methods
  static groupByMonth(dates) {
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const monthCounts = {};

    dates.forEach(date => {
      const monthKey = `${monthNames[date.getMonth()]} ${date.getFullYear()}`;
      monthCounts[monthKey] = (monthCounts[monthKey] || 0) + 1;
    });

    return monthCounts;
  }

  static countOccurrences(data, fieldName) {
    const counts = {};
    data.forEach(row => {
      const value = row[fieldName];
      if (value) {
        counts[value] = (counts[value] || 0) + 1;
      }
    });
    return counts;
  }

  static detectOutliers(values) {
    if (values.length < 4) return [];

    const sorted = [...values].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    return values.filter(v => v < lowerBound || v > upperBound);
  }

  static analyzeTimePatterns(data, timeField) {
    const trends = [];

    // Extract hour from time data for pattern analysis
    const times = data.map(row => {
      const timeStr = row[timeField.name];
      if (timeStr && typeof timeStr === 'string') {
        // Try to parse time patterns like "5:17:09 AM"
        const timeMatch = timeStr.match(/(\d{1,2}):(\d{2}).*?(AM|PM)/i);
        if (timeMatch) {
          let hour = parseInt(timeMatch[1]);
          const period = timeMatch[3].toUpperCase();
          if (period === 'PM' && hour !== 12) hour += 12;
          if (period === 'AM' && hour === 12) hour = 0;
          return hour;
        }
      }
      return null;
    }).filter(h => h !== null);

    if (times.length > 0) {
      // Find peak hour
      const hourCounts = {};
      times.forEach(hour => {
        hourCounts[hour] = (hourCounts[hour] || 0) + 1;
      });

      const peakHour = Object.entries(hourCounts)
        .sort(([,a], [,b]) => b - a)[0];

      if (peakHour) {
        const hour12 = peakHour[0] === 0 ? 12 : peakHour[0] > 12 ? peakHour[0] - 12 : peakHour[0];
        const period = peakHour[0] >= 12 ? 'PM' : 'AM';
        trends.push({
          type: 'time_pattern',
          message: `Peak activity at ${hour12}:00 ${period} (${peakHour[1]} records)`,
          icon: '🕐'
        });
      }
    }

    return trends;
  }

  static analyzeSalesCoverage(data, fields) {
    // Placeholder for sales-specific analysis
    return { description: 'Sales data analysis' };
  }

  static analyzePerformanceCoverage(data, fields) {
    // Placeholder for performance-specific analysis
    return { description: 'Performance data analysis' };
  }
}

export default IntelligentAnalyzer;