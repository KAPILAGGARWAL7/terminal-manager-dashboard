import React from 'react';

// Shared loading components for consistent UX
export const LoadingSpinner = ({ size = 'medium', color = '#3b82f6', message }) => {
    const sizeClasses = {
        small: 'w-4 h-4',
        medium: 'w-8 h-8',
        large: 'w-12 h-12'
    };

    return (
        <div className="loading-spinner-container">
            <div
                className={`loading-spinner ${sizeClasses[size]}`}
                style={{ borderTopColor: color }}
            />
            {message && <p className="loading-message">{message}</p>}

            <style jsx>{`
                .loading-spinner-container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 0.5rem;
                }

                .loading-spinner {
                    border: 2px solid #f3f4f6;
                    border-radius: 50%;
                    border-top-color: ${color};
                    animation: spin 1s ease-in-out infinite;
                }

                @keyframes spin {
                    to { transform: rotate(360deg); }
                }

                .loading-message {
                    color: #64748b;
                    font-size: 0.875rem;
                    margin: 0;
                    text-align: center;
                }
            `}</style>
        </div>
    );
};

export const LoadingCard = ({ title, description }) => (
    <div className="loading-card">
        <LoadingSpinner size="large" message="Processing..." />
        <div className="loading-card-content">
            {title && <h3>{title}</h3>}
            {description && <p>{description}</p>}
        </div>

        <style jsx>{`
            .loading-card {
                background: white;
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
                text-align: center;
                border: 1px solid #e5e7eb;
            }

            .loading-card-content {
                margin-top: 1rem;
            }

            .loading-card-content h3 {
                color: #1f2937;
                margin: 0 0 0.5rem 0;
                font-size: 1.125rem;
                font-weight: 600;
            }

            .loading-card-content p {
                color: #6b7280;
                margin: 0;
                font-size: 0.875rem;
            }
        `}</style>
    </div>
);

export const ProgressBar = ({ progress, label, showPercentage = true }) => (
    <div className="progress-container">
        {label && <p className="progress-label">{label}</p>}
        <div className="progress-bar">
            <div
                className="progress-fill"
                style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
        </div>
        {showPercentage && (
            <span className="progress-text">{Math.round(progress)}%</span>
        )}

        <style jsx>{`
            .progress-container {
                width: 100%;
            }

            .progress-label {
                color: #374151;
                font-size: 0.875rem;
                font-weight: 500;
                margin: 0 0 0.5rem 0;
            }

            .progress-bar {
                width: 100%;
                height: 8px;
                background: #f3f4f6;
                border-radius: 4px;
                overflow: hidden;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #3b82f6, #1d4ed8);
                border-radius: 4px;
                transition: width 0.3s ease;
            }

            .progress-text {
                color: #6b7280;
                font-size: 0.75rem;
                font-weight: 500;
            }
        `}</style>
    </div>
);

export default { LoadingSpinner, LoadingCard, ProgressBar };