import React from 'react';
import '../styles/TierComparisonTable.css';

/**
 * TierComparisonTable Component
 * Side-by-side comparison of all 3 quality tiers
 * 
 * Props:
 * - selectedTier: Currently selected tier
 * - onTierSelect: Callback when user clicks to select a tier
 * - tierSpecs: Array of tier specifications from API
 */
const TierComparisonTable = ({ selectedTier = 'good', onTierSelect, tierSpecs = [] }) => {
  const getTierColor = (tier) => {
    const colors = {
      good: '#2563eb',
      high: '#dc2626',
      ultra: '#a855f7',
    };
    return colors[tier];
  };

  const getTierIcon = (tier) => {
    const icons = {
      good: '📱',
      high: '🎬',
      ultra: '🎥',
    };
    return icons[tier];
  };

  const getTierLabel = (tier) => {
    const labels = {
      good: 'GOOD',
      high: 'HIGH',
      ultra: 'ULTRA',
    };
    return labels[tier];
  };

  const getTierUseCases = (tier) => {
    const useCases = {
      good: ['Social Media', 'YouTube', 'Instagram', 'TikTok'],
      high: ['Broadcast', 'Cinema', 'Premium Content', 'OTT Platforms'],
      ultra: ['4K HDR', 'Cinematic', 'Theatrical', 'Premium Cinemas'],
    };
    return useCases[tier];
  };

  const renderTierCard = (tier) => {
    const tierSpec = tierSpecs.find(t => t.tier === tier);
    if (!tierSpec) return null;

    return (
      <div
        key={tier}
        className={`tier-card ${selectedTier === tier ? 'selected' : ''}`}
        style={
          selectedTier === tier
            ? { borderColor: getTierColor(tier), boxShadow: `0 8px 24px ${getTierColor(tier)}40` }
            : {}
        }
        onClick={() => onTierSelect(tier)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            onTierSelect(tier);
          }
        }}
        aria-pressed={selectedTier === tier}
        aria-label={`${tier} tier: ${tierSpec?.quality_guarantee || ''}`}
      >
        {/* Header */}
        <div className="tier-card-header" style={{ backgroundColor: getTierColor(tier) }}>
          <div className="tier-icon-large">{getTierIcon(tier)}</div>
          <h3 className="tier-name-large">{getTierLabel(tier)}</h3>
          {selectedTier === tier && <div className="selection-badge">✓ Selected</div>}
        </div>

        {/* Body */}
        <div className="tier-card-body">
          {/* Guarantee */}
          <div className="guarantee-section">
            <p className="guarantee-text">{tierSpec?.quality_guarantee}</p>
          </div>

          {/* Key Specs */}
          <div className="specs-section">
            <div className="spec-row">
              <span className="spec-label">Resolution</span>
              <span className="spec-value">{tierSpec?.video_specs?.resolution}</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Frame Rate</span>
              <span className="spec-value">{tierSpec?.video_specs?.fps} fps</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Audio</span>
              <span className="spec-value">{tierSpec?.audio_specs?.format}</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Delivery</span>
              <span className="spec-value">{tierSpec?.delivery?.estimated_time_sec}s</span>
            </div>
          </div>

          {/* Use Cases */}
          <div className="use-cases-section">
            <p className="use-cases-label">Best For:</p>
            <div className="use-cases-tags">
              {getTierUseCases(tier).map((useCase, idx) => (
                <span key={idx} className="use-case-tag">
                  {useCase}
                </span>
              ))}
            </div>
          </div>

          {/* Pricing */}
          <div className="pricing-section">
            <p className="price-per-min">${tierSpec?.price_per_minute?.toFixed(2)}/min</p>
            <p className="pricing-note">
              {tier === 'good' && 'Professional quality at social media rates'}
              {tier === 'high' && 'Premium 4K broadcast quality'}
              {tier === 'ultra' && 'Maximum quality for cinematic delivery'}
            </p>
          </div>

          {/* Select Button */}
          <button
            className={`tier-select-btn ${selectedTier === tier ? 'selected' : ''}`}
            onClick={() => onTierSelect(tier)}
            style={
              selectedTier === tier
                ? { backgroundColor: getTierColor(tier), color: 'white' }
                : {
                    borderColor: getTierColor(tier),
                    color: getTierColor(tier),
                  }
            }
          >
            {selectedTier === tier ? '✓ Selected' : 'Select Tier'}
          </button>
        </div>

        {/* Certification Badge */}
        <div className="certification-badge">
          <span className="cert-icon">✓</span>
          <span className="cert-text">Certified Quality</span>
        </div>
      </div>
    );
  };

  return (
    <div className="tier-comparison-table">
      <div className="comparison-header">
        <h2>📊 Quality Tier Comparison</h2>
        <p className="comparison-subtitle">
          Compare features, quality guarantees, and pricing across all three tiers
        </p>
      </div>

      <div className="tiers-grid">
        {['good', 'high', 'ultra'].map((tier) => renderTierCard(tier))}
      </div>

      {/* Detailed Comparison Table */}
      <div className="detailed-comparison">
        <h3>Detailed Specifications</h3>
        <div className="comparison-table-wrapper">
          <table className="comparison-table">
            <thead>
              <tr>
                <th className="feature-column">Feature</th>
                <th className="tier-column good-column">GOOD</th>
                <th className="tier-column high-column">HIGH</th>
                <th className="tier-column ultra-column">ULTRA</th>
              </tr>
            </thead>
            <tbody>
              {/* Video Specs */}
              <tr className="section-header">
                <td colSpan="4">📹 Video Specifications</td>
              </tr>
              <tr>
                <td>Resolution</td>
                <td>{tierSpecs.find(t => t.tier === 'good')?.video_specs?.resolution}</td>
                <td>{tierSpecs.find(t => t.tier === 'high')?.video_specs?.resolution}</td>
                <td>{tierSpecs.find(t => t.tier === 'ultra')?.video_specs?.resolution}</td>
              </tr>
              <tr>
                <td>Frame Rate</td>
                <td>{tierSpecs.find(t => t.tier === 'good')?.video_specs?.fps} fps</td>
                <td>{tierSpecs.find(t => t.tier === 'high')?.video_specs?.fps} fps</td>
                <td>{tierSpecs.find(t => t.tier === 'ultra')?.video_specs?.fps} fps</td>
              </tr>
              <tr>
                <td>Codec</td>
                <td>{tierSpecs.find(t => t.tier === 'good')?.video_specs?.codec}</td>
                <td>{tierSpecs.find(t => t.tier === 'high')?.video_specs?.codec}</td>
                <td>{tierSpecs.find(t => t.tier === 'ultra')?.video_specs?.codec}</td>
              </tr>
              <tr>
                <td>Bitrate</td>
                <td>{tierSpecs.find(t => t.tier === 'good')?.video_specs?.bitrate_kbps} kbps</td>
                <td>{tierSpecs.find(t => t.tier === 'high')?.video_specs?.bitrate_kbps} kbps</td>
                <td>{tierSpecs.find(t => t.tier === 'ultra')?.video_specs?.bitrate_kbps} kbps</td>
              </tr>

              {/* Audio Specs */}
              <tr className="section-header">
                <td colSpan="4">🔊 Audio Specifications</td>
              </tr>
              <tr>
                <td>Format</td>
                <td>{tierSpecs.find(t => t.tier === 'good')?.audio_specs?.format}</td>
                <td>{tierSpecs.find(t => t.tier === 'high')?.audio_specs?.format}</td>
                <td>{tierSpecs.find(t => t.tier === 'ultra')?.audio_specs?.format}</td>
              </tr>
              <tr>
                <td>Channels</td>
                <td>{tierSpecs.find(t => t.tier === 'good')?.audio_specs?.channels} ch</td>
                <td>{tierSpecs.find(t => t.tier === 'high')?.audio_specs?.channels} ch</td>
                <td>{tierSpecs.find(t => t.tier === 'ultra')?.audio_specs?.channels} ch</td>
              </tr>
              <tr>
                <td>Loudness</td>
                <td>{tierSpecs.find(t => t.tier === 'good')?.audio_specs?.loudness_lufs} LUFS</td>
                <td>{tierSpecs.find(t => t.tier === 'high')?.audio_specs?.loudness_lufs} LUFS</td>
                <td>{tierSpecs.find(t => t.tier === 'ultra')?.audio_specs?.loudness_lufs} LUFS</td>
              </tr>

              {/* Delivery */}
              <tr className="section-header">
                <td colSpan="4">⏱️ Delivery & SLA</td>
              </tr>
              <tr>
                <td>Delivery Time</td>
                <td>{tierSpecs.find(t => t.tier === 'good')?.delivery?.estimated_time_sec}s</td>
                <td>{tierSpecs.find(t => t.tier === 'high')?.delivery?.estimated_time_sec}s</td>
                <td>{tierSpecs.find(t => t.tier === 'ultra')?.delivery?.estimated_time_sec}s</td>
              </tr>
              <tr>
                <td>Quality Guarantee</td>
                <td>✓ Certified</td>
                <td>✓ Certified</td>
                <td>✓ Certified</td>
              </tr>
              <tr>
                <td>Price</td>
                <td>${tierSpecs.find(t => t.tier === 'good')?.price_per_minute?.toFixed(2)}/min</td>
                <td>${tierSpecs.find(t => t.tier === 'high')?.price_per_minute?.toFixed(2)}/min</td>
                <td>${tierSpecs.find(t => t.tier === 'ultra')?.price_per_minute?.toFixed(2)}/min</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default TierComparisonTable;
