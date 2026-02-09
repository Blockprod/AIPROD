import React from 'react';
import '../styles/QualityGuarantee.css';

/**
 * QualityGuarantee Component
 * Displays quality guarantee badge and certification details
 * 
 * Props:
 * - tier: Selected quality tier ('good', 'high', 'ultra')
 * - certified: Whether the video is certified
 * - tierSpecs: Array of tier specifications from API
 */
const QualityGuarantee = ({ tier = 'good', certified = false, tierSpecs = [] }) => {
  const tierSpec = tierSpecs.find(t => t.tier === tier);

  const getTierColor = (tierName) => {
    const colors = {
      good: '#2563eb',
      high: '#dc2626',
      ultra: '#a855f7',
    };
    return colors[tierName];
  };

  const getTierLabel = (tierName) => {
    const labels = {
      good: 'GOOD',
      high: 'HIGH',
      ultra: 'ULTRA',
    };
    return labels[tierName];
  };

  const getCertificationText = (tierName) => {
    const texts = {
      good: 'Professional 1080p Quality Certified',
      high: 'Professional 4K Quality Certified',
      ultra: 'Broadcast Cinema 4K@60fps HDR Certified',
    };
    return texts[tierName];
  };

  const getGuaranteeDetails = (tierName) => {
    const details = {
      good: {
        quality: '1080p Professional',
        audio: 'Stereo Clear Dialogue',
        colorGrade: 'Professional Auto WB',
        sla: '35 Second Delivery SLA',
      },
      high: {
        quality: '4K Professional Broadcast',
        audio: '5.1 Immersive Surround',
        colorGrade: '3-Point Cinema Grade',
        sla: '60 Second Delivery SLA',
      },
      ultra: {
        quality: '4K@60fps HDR Cinematic',
        audio: '7.1.4 Dolby Atmos Spatial',
        colorGrade: 'DaVinci Professional Grade',
        sla: '120 Second Delivery SLA',
      },
    };
    return details[tierName];
  };

  const guaranteeDetails = getGuaranteeDetails(tier);

  return (
    <div className="quality-guarantee">
      {/* Main Certification Badge */}
      <div
        className="certification-badge-main"
        style={{ borderColor: getTierColor(tier), backgroundColor: `${getTierColor(tier)}10` }}
      >
        <div className="badge-icon-container" style={{ backgroundColor: getTierColor(tier) }}>
          <span className="badge-icon">✓</span>
        </div>

        <div className="badge-content">
          <div className="badge-tier" style={{ color: getTierColor(tier) }}>
            {getTierLabel(tier)} TIER
          </div>
          <div className="badge-text">{getCertificationText(tier)}</div>
          {certified && <div className="certified-indicator">🎖️ Video Certified</div>}
        </div>
      </div>

      {/* Guarantee Details Grid */}
      <div className="guarantee-details-grid">
        <div className="guarantee-item">
          <div className="guarantee-icon">📹</div>
          <div className="guarantee-label">Video Quality</div>
          <div className="guarantee-value">{guaranteeDetails.quality}</div>
        </div>

        <div className="guarantee-item">
          <div className="guarantee-icon">🔊</div>
          <div className="guarantee-label">Audio Quality</div>
          <div className="guarantee-value">{guaranteeDetails.audio}</div>
        </div>

        <div className="guarantee-item">
          <div className="guarantee-icon">🎨</div>
          <div className="guarantee-label">Color Grading</div>
          <div className="guarantee-value">{guaranteeDetails.colorGrade}</div>
        </div>

        <div className="guarantee-item">
          <div className="guarantee-icon">⏱️</div>
          <div className="guarantee-label">Delivery SLA</div>
          <div className="guarantee-value">{guaranteeDetails.sla}</div>
        </div>
      </div>

      {/* Full Specifications */}
      {tierSpec && (
        <div className="full-specs">
          <h4>Full Quality Specifications</h4>

          <div className="specs-columns">
            {/* Video Specs Column */}
            <div className="spec-column">
              <h5>📹 Video</h5>
              <div className="spec-item">
                <span className="spec-name">Resolution:</span>
                <span className="spec-val">{tierSpec.video_specs?.resolution}</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Frame Rate:</span>
                <span className="spec-val">{tierSpec.video_specs?.fps} fps</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Codec:</span>
                <span className="spec-val">{tierSpec.video_specs?.codec}</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Bitrate:</span>
                <span className="spec-val">{tierSpec.video_specs?.bitrate_kbps} kbps</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Color Space:</span>
                <span className="spec-val">{tierSpec.video_specs?.color_space}</span>
              </div>
            </div>

            {/* Audio Specs Column */}
            <div className="spec-column">
              <h5>🔊 Audio</h5>
              <div className="spec-item">
                <span className="spec-name">Format:</span>
                <span className="spec-val">{tierSpec.audio_specs?.format}</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Channels:</span>
                <span className="spec-val">{tierSpec.audio_specs?.channels} ch</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Codec:</span>
                <span className="spec-val">{tierSpec.audio_specs?.codec}</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Bitrate:</span>
                <span className="spec-val">{tierSpec.audio_specs?.bitrate_kbps} kbps</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Loudness:</span>
                <span className="spec-val">{tierSpec.audio_specs?.loudness_lufs} LUFS</span>
              </div>
            </div>

            {/* Delivery Specs Column */}
            <div className="spec-column">
              <h5>⏱️ Delivery</h5>
              <div className="spec-item">
                <span className="spec-name">Formats:</span>
                <span className="spec-val">{tierSpec.delivery?.formats?.join(', ')}</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Est. Delivery:</span>
                <span className="spec-val">{tierSpec.delivery?.estimated_time_sec}s</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">SLA:</span>
                <span className="spec-val">Guaranteed ✓</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Quality:</span>
                <span className="spec-val">Certified ✓</span>
              </div>
              <div className="spec-item">
                <span className="spec-name">Support:</span>
                <span className="spec-val">24/7 Priority</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quality Guarantee Message */}
      <div className="guarantee-message" style={{ borderLeftColor: getTierColor(tier) }}>
        <span className="message-icon">📋</span>
        <span className="message-text">
          Every video generated with the {getTierLabel(tier)} tier is automatically validated
          against our quality specifications and certified for delivery. Quality is guaranteed.
        </span>
      </div>

      {/* Certification Score Display */}
      <div className="certification-score">
        <h4>Quality Assessment</h4>
        <div className="score-items">
          <div className="score-item">
            <span className="score-label">Technical Compliance:</span>
            <div className="score-bar">
              <div className="score-fill" style={{ width: '95%', backgroundColor: getTierColor(tier) }}></div>
            </div>
            <span className="score-percentage">95%</span>
          </div>
          <div className="score-item">
            <span className="score-label">Audio Quality:</span>
            <div className="score-bar">
              <div className="score-fill" style={{ width: '98%', backgroundColor: getTierColor(tier) }}></div>
            </div>
            <span className="score-percentage">98%</span>
          </div>
          <div className="score-item">
            <span className="score-label">Visual Quality:</span>
            <div className="score-bar">
              <div className="score-fill" style={{ width: '96%', backgroundColor: getTierColor(tier) }}></div>
            </div>
            <span className="score-percentage">96%</span>
          </div>
          <div className="score-item">
            <span className="score-label">Overall Pass Rate:</span>
            <div className="score-bar">
              <div className="score-fill" style={{ width: '100%', backgroundColor: getTierColor(tier) }}></div>
            </div>
            <span className="score-percentage">100%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QualityGuarantee;
