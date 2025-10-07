import React from 'react';
import './ThemeToggle.css';

const ThemeToggle = ({ isDark, onToggle }) => {
  return (
    <div className="theme-toggle-container">
      <button
        className={`theme-toggle ${isDark ? 'dark' : 'light'}`}
        onClick={onToggle}
        aria-label="Toggle theme"
      >
        <div className="toggle-slider">
          <span className="toggle-icon">
            {isDark ? '🌙' : '☀️'}
          </span>
        </div>
      </button>
    </div>
  );
};

export default ThemeToggle;