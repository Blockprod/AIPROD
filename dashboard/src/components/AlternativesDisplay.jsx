import React, { useState, useEffect } from 'react';
import '../styles/AlternativesDisplay.css';

/**
 * AlternativesDisplay Component
 * Shows cost comparison for all 3 tiers at once
 * 
 * Props:
 * - selectedTier: Currently selected tier
 * - duration: Duration in seconds
 * - complexity: Complexity level
 * - batchCount: Number of videos in batch
 * - getAllAlternatives: Function to fetch alternatives from API
 * - loading: Loading state
 */
const AlternativesDisplay = ({
  selectedTier = 'good',
  duration = 60,
  complexity = 'moderate',
  batchCount = 1,
  getAllAlternatives,
  loading = false,
}) => {
  const [alternatives, setAlternatives] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);

  // Load alternatives when parameters change
  useEffect(() => {
    if (getAllAlternatives) {
      getAllAlternatives({
        duration_sec: duration,
        complexity,
        batch_count: batchCount,
      }).then(setAlternatives);
    }
  }, [duration, complexity, batchCount, getAllAlternatives]);

  const getTierColor = (tier) => {
    const colors = {
      good: '#2563eb',
      high: '#dc2626',
      ultra: '#a855f7',
    };
    return colors[tier];
  };

  const getTierLabel = (tier) => {
    const labels = {
      good: 'GOOD',
      high: 'HIGH',
      ultra: 'ULTRA',
    };
    return labels[tier];
  };

  const getSavingsInfo = (selectedCost, altCost) => {
    if (!selectedCost || !altCost) return null;
    const selected = parseFloat(selectedCost.total?.replace('$', '') || 0);
    const alt = parseFloat(altCost.total?.replace('$', '') || 0);
    const diff = selected - alt;
    const percent = ((diff / selected) * 100).toFixed(0);
    return { diff: Math.abs(diff), percent: Math.abs(percent), more: diff < 0 };
  };

  const renderAltCard = (tier, costData) => {
    if (!costData) return null;

    const isSelected = tier === selectedTier;
    const color = getTierColor(tier);
    const selected = alternatives?.[selectedTier];
    const savings = selected && !isSelected ? getSavingsInfo(selected, costData) : null;

    return (
      <div
        key={tier}
        className={`alternative-card ${isSelected ? 'selected' : ''}`}
        style={
          isSelected
            ? {
                borderColor: color,
                backgroundColor: `${color}10`,
                boxShadow: `0 8px 24px ${color}30`,
              }
            : {}
        }
      >
        {/* Header */}
        <div className="alt-header" style={{ borderBottomColor: color }}>
          <div className="alt-tier-name" style={{ color }}>
            {getTierLabel(tier)}
          </div>
          {isSelected && <span className="alt-selected-badge">✓ Selected</span>}
        </div>

        {/* Price Display */}
        <div className="alt-price" style={{ color }}>
          <div className="price-value">
            {costData.total ? costData.total.replace('$', '') : '$0.00'}
          </div>
          <div className="price-label">Total Cost</div>
        </div>

        {/* Savings Indicator */}
        {savings && !isSelected && (
          <div
            className={`savings-indicator ${savings.more ? 'more' : 'less'}`}
            style={{
              backgroundColor: savings.more ? '#ef4444' : '#10b981',
              borderColor: savings.more ? '#991b1b' : '#065f46',
            }}
          >
            <span className="savings-icon">{savings.more ? '📈' : '📉'}</span>
            <span className="savings-text">
              {savings.more ? '+' : '-'}${savings.diff.toFixed(2)} ({savings.percent}%)
            </span>
          </div>
        )}

        {/* Breakdown Preview */}
        <div className="alt-breakdown-preview">
          <div className="breakdown-item">
            <span className="breakdown-label">Base:</span>
            <span className="breakdown-value">${costData.base_cost?.toFixed(3)}</span>
          </div>
          <div className="breakdown-item">
            <span className="breakdown-label">With multipliers:</span>
            <span className="breakdown-value">${costData.with_batch?.toFixed(3)}</span>
          </div>
          <div className="breakdown-item">
            <span className="breakdown-label">With tax:</span>
            <span className="breakdown-value">{costData.total}</span>
          </div>
        </div>

        {/* Delivery Time */}
        <div className="alt-delivery">
          <span className="delivery-icon">⏱️</span>
          <span className="delivery-time">{costData.estimated_delivery_sec}s delivery</span>
        </div>
      </div>
    );
  };

  return (
    <div className="alternatives-display">
      <div className="alternatives-header">
        <h3>💡 All Tier Options</h3>
        <button
          className="expand-button"
          onClick={() => setIsExpanded(!isExpanded)}
          aria-expanded={isExpanded}
        >
          {isExpanded ? '▼ Hide Alternatives' : '▶ Show All Options'}
        </button>
      </div>

      {isExpanded && (
        <div className="alternatives-container">
          {loading && (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Loading tier alternatives...</p>
            </div>
          )}

          {alternatives && !loading && (
            <>
              <div className="alternatives-grid">
                {['good', 'high', 'ultra'].map((tier) =>
                  renderAltCard(tier, alternatives[tier])
                )}
              </div>

              {/* Cost Comparison Summary */}
              <div className="comparison-summary">
                <h4>Cost Summary for {duration}s {complexity} video ({batchCount} video{batchCount !== 1 ? 's' : ''})</h4>
                <div className="summary-table">
                  <div className="summary-row header">
                    <div className="summary-col tier-col">Tier</div>
                    <div className="summary-col cost-col">Cost</div>
                    <div className="summary-col delivery-col">Delivery</div>
                    <div className="summary-col quality-col">Quality</div>
                  </div>

                  {['good', 'high', 'ultra'].map((tier) => (
                    <div
                      key={`summary-${tier}`}
                      className={`summary-row ${selectedTier === tier ? 'selected' : ''}`}
                    >
                      <div className="summary-col tier-col" style={{ color: getTierColor(tier) }}>
                        {getTierLabel(tier)}
                      </div>
                      <div className="summary-col cost-col">
                        <strong>{alternatives[tier]?.total}</strong>
                      </div>
                      <div className="summary-col delivery-col">
                        {alternatives[tier]?.estimated_delivery_sec}s
                      </div>
                      <div className="summary-col quality-col">
                        {tier === 'good' && '1080p'}
                        {tier === 'high' && '4K'}
                        {tier === 'ultra' && '4K@60 HDR'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Value Proposition */}
              <div className="value-proposition">
                <h4>Which tier is right for you?</h4>
                <div className="value-boxes">
                  <div className="value-box">
                    <h5>GOOD Tier</h5>
                    <p>Best for social media creators, YouTube, TikTok, and Instagram content</p>
                    <ul>
                      <li>Professional 1080p quality</li>
                      <li>Clear stereo dialogue</li>
                      <li>Quick 35-second delivery</li>
                      <li>Most affordable option</li>
                    </ul>
                  </div>

                  <div className="value-box">
                    <h5>HIGH Tier</h5>
                    <p>Ideal for broadcast, streaming platforms, and premium content</p>
                    <ul>
                      <li>4K professional broadcast</li>
                      <li>5.1 immersive audio</li>
                      <li>Cinema-grade color</li>
                      <li>Best balance of quality and price</li>
                    </ul>
                  </div>

                  <div className="value-box">
                    <h5>ULTRA Tier</h5>
                    <p>Perfect for theatrical, cinema, and highest-end productions</p>
                    <ul>
                      <li>4K@60fps HDR cinematic</li>
                      <li>7.1.4 Dolby Atmos</li>
                      <li>DaVinci professional color</li>
                      <li>Maximum quality guarantee</li>
                    </ul>
                  </div>
                </div>
              </div>
            </>
          )}

          {!alternatives && !loading && (
            <div className="empty-state">
              <p>Select parameters to view tier alternatives</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AlternativesDisplay;
