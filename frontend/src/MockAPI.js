// Mock REST API to simulate Terminal Manager data
export class MockTerminalAPI {
    static async getTerminalSites() {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1500));

        return {
            success: true,
            data: [
                { siteId: 'TM001', siteName: 'Downtown Terminal', fuelType: 'Diesel', dailyVolume: 15000, region: 'North', operatingHours: 24, status: 'Active' },
                { siteId: 'TM002', siteName: 'Airport Hub', fuelType: 'Jet Fuel', dailyVolume: 25000, region: 'South', operatingHours: 24, status: 'Active' },
                { siteId: 'TM003', siteName: 'Port Terminal', fuelType: 'Marine Gas', dailyVolume: 18000, region: 'East', operatingHours: 16, status: 'Maintenance' },
                { siteId: 'TM004', siteName: 'Industrial Park', fuelType: 'Diesel', dailyVolume: 12000, region: 'West', operatingHours: 12, status: 'Active' },
                { siteId: 'TM005', siteName: 'Highway Station', fuelType: 'Gasoline', dailyVolume: 8000, region: 'North', operatingHours: 24, status: 'Active' },
                { siteId: 'TM006', siteName: 'Central Depot', fuelType: 'Diesel', dailyVolume: 22000, region: 'South', operatingHours: 20, status: 'Active' },
                { siteId: 'TM007', siteName: 'Coastal Terminal', fuelType: 'Marine Gas', dailyVolume: 16000, region: 'East', operatingHours: 18, status: 'Active' },
                { siteId: 'TM008', siteName: 'Mountain Base', fuelType: 'Gasoline', dailyVolume: 9500, region: 'West', operatingHours: 14, status: 'Active' },
                { siteId: 'TM009', siteName: 'City Center', fuelType: 'Diesel', dailyVolume: 13500, region: 'North', operatingHours: 24, status: 'Active' },
                { siteId: 'TM010', siteName: 'Suburban Hub', fuelType: 'Gasoline', dailyVolume: 11000, region: 'South', operatingHours: 16, status: 'Active' }
            ]
        };
    }

    static async getSalesAnalytics() {
        await new Promise(resolve => setTimeout(resolve, 1200));

        return {
            success: true,
            data: [
                { month: 'Jan 2024', totalSales: 450000, region: 'North', fuelType: 'Diesel', profit: 45000 },
                { month: 'Feb 2024', totalSales: 520000, region: 'North', fuelType: 'Diesel', profit: 52000 },
                { month: 'Mar 2024', totalSales: 480000, region: 'North', fuelType: 'Diesel', profit: 48000 },
                { month: 'Jan 2024', totalSales: 380000, region: 'South', fuelType: 'Jet Fuel', profit: 76000 },
                { month: 'Feb 2024', totalSales: 410000, region: 'South', fuelType: 'Jet Fuel', profit: 82000 },
                { month: 'Mar 2024', totalSales: 425000, region: 'South', fuelType: 'Jet Fuel', profit: 85000 },
                { month: 'Jan 2024', totalSales: 320000, region: 'East', fuelType: 'Marine Gas', profit: 32000 },
                { month: 'Feb 2024', totalSales: 350000, region: 'East', fuelType: 'Marine Gas', profit: 35000 },
                { month: 'Mar 2024', totalSales: 340000, region: 'East', fuelType: 'Marine Gas', profit: 34000 },
                { month: 'Jan 2024', totalSales: 280000, region: 'West', fuelType: 'Gasoline', profit: 28000 },
                { month: 'Feb 2024', totalSales: 310000, region: 'West', fuelType: 'Gasoline', profit: 31000 },
                { month: 'Mar 2024', totalSales: 295000, region: 'West', fuelType: 'Gasoline', profit: 29500 }
            ]
        };
    }

    static async getPerformanceMetrics() {
        await new Promise(resolve => setTimeout(resolve, 800));

        return {
            success: true,
            data: [
                { terminal: 'Downtown Terminal', efficiency: 94.5, uptime: 99.2, throughput: 1500, quality: 98.1 },
                { terminal: 'Airport Hub', efficiency: 96.8, uptime: 99.8, throughput: 2100, quality: 99.1 },
                { terminal: 'Port Terminal', efficiency: 89.2, uptime: 95.5, throughput: 1800, quality: 97.3 },
                { terminal: 'Industrial Park', efficiency: 91.7, uptime: 98.1, throughput: 1200, quality: 96.8 },
                { terminal: 'Highway Station', efficiency: 93.3, uptime: 97.9, throughput: 800, quality: 97.5 },
                { terminal: 'Central Depot', efficiency: 97.1, uptime: 99.5, throughput: 2200, quality: 99.3 },
                { terminal: 'Coastal Terminal', efficiency: 92.4, uptime: 96.7, throughput: 1600, quality: 98.0 },
                { terminal: 'Mountain Base', efficiency: 88.9, uptime: 94.2, throughput: 950, quality: 95.8 }
            ]
        };
    }
}

export const mockAPIEndpoints = {
    '/api/terminals/sites': MockTerminalAPI.getTerminalSites,
    '/api/analytics/sales': MockTerminalAPI.getSalesAnalytics,
    '/api/metrics/performance': MockTerminalAPI.getPerformanceMetrics
};