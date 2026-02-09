import React from 'react';
import '../styles/QualityTierSelector.css';

/**
 * QualityTierSelector Component
 * Displays 3 quality tier options with descriptions and selection
 * 
 * Props:
 * - selectedTier: Currently selected tier ('good', 'high', 'ultra')
 * - onTierChange: Callback when tier is selected
 * - tierSpecs: Array of tier specifications from API
 */
const QualityTierSelector = ({ selectedTier = 'good', onTierChange, tierSpecs = [] }) => {
  const getTierIcon = (tier) => {
    const icons = {
      good: '📱',
      high: '🎬',
      ultra: '🎥',
    };
    return icons[tier] || '▪';
  };

  const getTierDescription = (tier) => {
    const descriptions = {
      good: 'Social Media Professional - 1080p professional quality for content creators',
      high: 'Professional Broadcast - 4K cinema-grade quality with surround audio',
      ultra: 'Cinematic Excellence - 4K@60fps HDR with immersive Atmos audio',
    };
    return descriptions[tier];
  };

  const getTierColor = (tier) => {
    const colors = {
      good: '#2563eb', // Blue
      high: '#dc2626', // Red
      ultra: '#a855f7', // Purple
    };
    return colors[tier];
  };

  return (
    <div className="quality-tier-selector">
      <div className="tier-selector-header">
        <h2>Choose Your Quality Tier</h2>
        <p className="tier-selector-subtitle">
          Professional quality guaranteed. Select the tier that matches your needs.
        </p>
      </div>

      <div className="tier-buttons-container">
        {['good', 'high', 'ultra'].map((tier) => (
          <button
            key={tier}
            className={`tier-button tier-button-${tier} ${selectedTier === tier ? 'active' : ''}`}
            onClick={() => onTierChange(tier)}
            style={
              selectedTier === tier
                ? { borderColor: getTierColor(tier), boxShadow: `0 0 20px ${getTierColor(tier)}40` }
                : {}
            }
            aria-pressed={selectedTier === tier}
            aria-label={`Select ${tier} tier`}
          >
            <div className="tier-icon">{getTierIcon(tier)}</div>
            <div className="tier-content">
              <div className="tier-name">
                {tier === 'good' && '📱 GOOD'}
                {tier === 'high' && '🎬 HIGH'}
                {tier === 'ultra' && '🎥 ULTRA'}
              </div>
              <div className="tier-label">
                {tier === 'good' && 'Social Media Pro'}
                {tier === 'high' && 'Professional 4K'}
                {tier === 'ultra' && 'Cinematic HDR'}
              </div>
            </div>
            {selectedTier === tier && <div className="tier-check">✓</div>}
          </button>
        ))}
      </div>

      {/* Detailed tier descriptions */}
      <div className="tier-details">
        {['good', 'high', 'ultra'].map((tier) => (
          <div
            key={`details-${tier}`}
            className={`tier-detail-card ${selectedTier === tier ? 'visible' : 'hidden'}`}
          >
            <h3>{getTierDescription(tier)}</h3>
            
            {tierSpecs.length > 0 && (
              <div className="tier-specs-grid">
                {tierSpecs.find(t => t.tier === tier) && (
                  <>
                    {/* Video Specs */}
                    <div className="spec-group">
                      <h4>📹 Video</h4>
                      <ul>
                        {tierSpecs
                          .find(t => t.tier === tier)
                          ?.video_specs && (
                          <>
                            <li>
                              Resolution:{' '}
                              <strong>
                                {tierSpecs.find(t => t.tier === tier).video_specs.resolution}
                              </strong>
                            </li>
                            <li>
                              Frame Rate:{' '}
                              <strong>
                                {tierSpecs.find(t => t.tier === tier).video_specs.fps} fps
                              </strong>
                            </li>
                            <li>
                              Codec:{' '}
                              <strong>
                                {tierSpecs.find(t => t.tier === tier).video_specs.codec}
                              </strong>
                            </li>
                            <li>
                              Bitrate:{' '}
                              <strong>
                                {tierSpecs.find(t => t.tier === tier).video_specs.bitrate_kbps}{' '}
                                kbps
                              </strong>
                            </li>
                          </>
                        )}
                      </ul>
                    </div>

                    {/* Audio Specs */}
                    <div className="spec-group">
                      <h4>🔊 Audio</h4>
                      <ul>
                        {tierSpecs
                          .find(t => t.tier === tier)
                          ?.audio_specs && (
                          <>
                            <li>
                              Format:{' '}
                              <strong>
                                {tierSpecs.find(t => t.tier === tier).audio_specs.format}
                              </strong>
                            </li>
                            <li>
                              Channels:{' '}
                              <strong>
                                {tierSpecs.find(t => t.tier === tier).audio_specs.channels}{' '}
                                ch
                              </strong>
                            </li>
                            <li>
                              Codec:{' '}
                              <strong>
                                {tierSpecs.find(t => t.tier === tier).audio_specs.codec}
                              </strong>
                            </li>
                            <li>
                              Loudness:{' '}
                              <strong>
                                {tierSpecs.find(t => t.tier === tier).audio_specs.loudness_lufs}{' '}
                                LUFS
                              </strong>
                            </li>
                          </>
                        )}
                      </ul>
                    </div>

                    {/* Delivery Specs */}
                    <div className="spec-group">
                      <h4>⏱️ Delivery</h4>
                      <ul>
                        {tierSpecs
                          .find(t => t.tier === tier)
                          ?.delivery && (
                          <>
                            <li>
                              Formats:{' '}
                              <strong>
                                {tierSpecs.find(t => t.tier === tier).delivery.formats.join(', ')}
                              </strong>
                            </li>
                            <li>
                              Standard Delivery:{' '}
                              <strong>
                                {tierSpecs.find(t => t.tier === tier).delivery.estimated_time_sec}s
                              </strong>
                            </li>
                            <li>
                              SLA: <strong>Guaranteed</strong>
                            </li>
                            <li>
                              Quality: <strong>Certified ✓</strong>
                            </li>
                          </>
                        )}
                      </ul>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default QualityTierSelector;
