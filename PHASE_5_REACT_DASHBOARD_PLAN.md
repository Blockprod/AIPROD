# AIPROD Quality First - Phase 5: React Dashboard Update

**Status**: PLANNING PHASE  
**Objective**: Expose Quality First framework to end users via React dashboard  
**Estimated Duration**: 4-6 hours

---

## Overview

Phase 5 focuses on updating the React dashboard UI to showcase the Quality First model with:

- Modern tier selection interface (GOOD/HIGH/ULTRA)
- Real-time cost calculator display
- Quality guarantee certification badges
- Professional tier comparison table
- Visual quality tier descriptions

---

## Component Architecture

### 1. QualityTierSelector Component

**Purpose**: Allow users to select quality tier

**Props**:

```typescript
interface QualityTierSelectorProps {
  selectedTier: "good" | "high" | "ultra";
  onTierChange: (tier: "good" | "high" | "ultra") => void;
  tierSpecs: TierSpec[];
}
```

**Features**:

- 3 tier buttons with visual distinction
- Hover tooltips showing key specs
- Visual quality indicators (stars, badges)
- Tier descriptions from API

**Styling**:

- GOOD: Blue theme, social media icon
- HIGH: Professional broadcast colors (red/gold)
- ULTRA: Premium HDR/cinematic styling (purple/gold)

---

### 2. CostEstimator Component

**Purpose**: Real-time cost calculation display

**Props**:

```typescript
interface CostEstimatorProps {
  duration: number;
  selectedTier: "good" | "high" | "ultra";
  complexity: "simple" | "moderate" | "complex";
  rushDelivery: "standard" | "express_6h" | "express_2h" | "on_demand";
  batchCount: number;
}
```

**Features**:

- Real-time cost updates as inputs change
- Cost breakdown visualization (stacked bar chart)
- Base cost → complexity → rush → batch display
- Tax calculation shown
- Delivery time estimate

**API Call**:

```bash
POST /quality/estimate
Content-Type: application/json

{
  "tier": "high",
  "duration_sec": 60,
  "complexity": "moderate",
  "rush_delivery": "standard",
  "batch_count": 1
}
```

**Response**:

```json
{
  "tier_name": "high",
  "base_cost": 0.15,
  "complexity_adjusted": 0.18,
  "with_rush": 0.18,
  "with_batch": 0.18,
  "total_usd": 0.19,
  "estimated_delivery_sec": 60
}
```

---

### 3. QualityGuarantee Component

**Purpose**: Display quality guarantee badge and specs

**Props**:

```typescript
interface QualityGuaranteeProps {
  tier: "good" | "high" | "ultra";
  certified: boolean;
}
```

**Features**:

- Professional certification badge
- Key specs quick-view (resolution, fps, audio)
- Delivery time SLA display
- Copy-paste guarantee text
- "Certified Quality" seal animation

---

### 4. TierComparisionTable Component

**Purpose**: Side-by-side tier comparison

**Display**:

```
                GOOD              HIGH              ULTRA
Resolution      1080p@24fps       4K@30fps          4K@60fps HDR
Audio           Stereo 2.0        5.1 Surround      7.1.4 Atmos
Delivery        35 seconds        60 seconds        120 seconds
Price/min       $0.05             $0.15             $0.75
Use Case        Social media      Broadcast         Cinematic
Certification   ✅                ✅                ✅
```

**Features**:

- Responsive table
- Highlight comparison row
- Click to select tier
- Mobile-friendly view (vertical stack)

---

### 5. AlternativesDisplay Component

**Purpose**: Show cost for all 3 tiers at once

**API Call**:

```bash
POST /quality/estimate?show_alternatives=true
```

**Display**:

```
Duration: 60 seconds
Complexity: moderate

GOOD:  $0.03 per minute
HIGH:  $0.08 per minute
ULTRA: $0.41 per minute
```

---

## Integration Points

### API Endpoints Used

1. **GET /quality/tiers**
   - Called on page load
   - Populates tier selector options
   - Shows quality guarantees

2. **POST /quality/estimate**
   - Called on every input change (debounced)
   - Updates cost display real-time
   - Shows tier alternatives

3. **POST /quality/validate** (Optional, for file upload)
   - Called after video generated
   - Shows QC report
   - Displays certification status

---

## State Management

### Redux Store Structure

```typescript
interface QualityState {
  selectedTier: "good" | "high" | "ultra";
  duration: number;
  complexity: "simple" | "moderate" | "complex";
  rushDelivery: "standard" | "express_6h" | "express_2h" | "on_demand";
  batchCount: number;
  currentCost: CostBreakdown | null;
  allTierSpecs: TierSpec[];
  loading: boolean;
  error: string | null;
}

interface CostBreakdown {
  tier_name: string;
  base_cost: number;
  complexity_adjusted: number;
  with_rush: number;
  with_batch: number;
  total_usd: number;
  estimated_delivery_sec: number;
}
```

### Actions

- `setSelectedTier(tier)`
- `setDuration(seconds)`
- `setComplexity(level)`
- `setRushDelivery(option)`
- `setBatchCount(count)`
- `updateCostEstimate()`
- `loadTierSpecs()`

---

## UI/UX Specifications

### Tier Selector Button Styling

**GOOD Tier Button**:

```css
.tier-button-good {
  background: linear-gradient(135deg, #2563eb, #1e40af);
  border: 2px solid #1e40af;
  color: white;
  font-weight: bold;
  padding: 16px 24px;
  border-radius: 8px;
}

.tier-button-good:hover {
  transform: scale(1.05);
  box-shadow: 0 8px 16px rgba(37, 99, 235, 0.3);
}

.tier-button-good.active {
  border-color: #ffffff;
  box-shadow: 0 0 20px rgba(37, 99, 235, 0.6);
}
```

**HIGH Tier Button**:

```css
.tier-button-high {
  background: linear-gradient(135deg, #dc2626, #991b1b);
  border: 2px solid #991b1b;
  color: white;
  font-weight: bold;
}
```

**ULTRA Tier Button**:

```css
.tier-button-ultra {
  background: linear-gradient(135deg, #a855f7, #7e22ce);
  border: 2px solid #7e22ce;
  color: white;
  font-weight: bold;
}
```

---

## Implementation Checklist

### Week 1: Planning & Design

- [ ] Design Figma mockups (tier buttons, cost display, comparison)
- [ ] Get stakeholder approval on tier positioning
- [ ] Document component specifications
- [ ] Plan API integration flow

### Week 2: Component Implementation

- [ ] Create QualityTierSelector component
- [ ] Implement CostEstimator with real-time updates
- [ ] Build QualityGuarantee badge component
- [ ] Create TierComparisonTable

### Week 3: API Integration & Testing

- [ ] Integrate GET /quality/tiers endpoint
- [ ] Integrate POST /quality/estimate endpoint
- [ ] Implement debounced cost updates
- [ ] Add error handling for API calls
- [ ] Test all edge cases (invalid tiers, network errors)

### Week 4: Polish & Launch

- [ ] Responsive design testing (mobile, tablet, desktop)
- [ ] Accessibility review (WCAG 2.1 AA)
- [ ] Performance optimization (lazy loading, memoization)
- [ ] User acceptance testing
- [ ] Documentation updates
- [ ] Deploy to production

---

## File Changes Needed

### New React Components

1. `src/components/QualityTierSelector.tsx` (~150 LOC)
2. `src/components/CostEstimator.tsx` (~200 LOC)
3. `src/components/QualityGuarantee.tsx` (~120 LOC)
4. `src/components/TierComparisonTable.tsx` (~180 LOC)
5. `src/components/AlternativesDisplay.tsx` (~100 LOC)

### New Hooks

6. `src/hooks/useQualityCalculator.ts` (~80 LOC)
7. `src/hooks/useTierSpecs.ts` (~60 LOC)

### Styles

8. `src/styles/quality-tiers.css` (~200 LOC)
9. `src/styles/cost-estimator.css` (~150 LOC)

### Redux/State

10. `src/store/quality.slice.ts` (~150 LOC)

### Services/API

11. `src/services/qualityApi.ts` (~100 LOC)

### Updated Files

12. `src/pages/VideoGenerator.tsx` - Integration point
13. `src/layouts/MainLayout.tsx` - If adding to dashboard

**Total New LOC**: ~1,500
**Total Files**: 13

---

## Testing Strategy

### Unit Tests

```typescript
// QualityTierSelector.test.tsx
describe("QualityTierSelector", () => {
  it("renders all 3 tier options");
  it("calls onTierChange when tier selected");
  it("displays tier descriptions from props");
  it("Shows active state for selected tier");
});

// CostEstimator.test.tsx
describe("CostEstimator", () => {
  it("displays correct cost for given inputs");
  it("updates cost in real-time on duration change");
  it("applies complexity multiplier correctly");
  it("shows delivery time estimate");
  it("handles API errors gracefully");
});
```

### Integration Tests

```typescript
describe("Quality First Flow", () => {
  it("loads tier specs on component mount");
  it("estimates cost for each tier option");
  it("tier selection updates cost display");
  it("batch count applies correct discount");
  it("rush delivery multiplier applied");
});
```

### E2E Tests (Cypress)

```typescript
describe("Quality First Dashboard", () => {
  it("User selects HIGH tier and sees updated cost");
  it("User changes duration and cost updates");
  it("User selects rush delivery and cost increases");
  it("User sees all tier alternatives");
});
```

---

## Performance Considerations

### Optimization Strategies

1. **Debounce Cost Updates**: Delay API call 300ms after user stops typing
2. **Memoize Components**: Prevent unnecessary re-renders with React.memo()
3. **Lazy Load Tier Specs**: Load on first interaction, cache result
4. **Optimize Bundle**: Tree-shake unused utility functions
5. **Image Optimization**: Progressive JPEG for tier icons

### Expected Performance Metrics

- Page load: <2 seconds
- Cost calculation response: <100ms
- UI interaction responsiveness: <16ms (60 fps)

---

## Accessibility Requirements

### WCAG 2.1 AA Compliance

- [ ] Keyboard navigation for all controls
- [ ] Color contrast ratios ≥4.5:1 for text
- [ ] Semantic HTML structure
- [ ] ARIA labels for custom components
- [ ] Screen reader testing with NVDA/JAWS
- [ ] Focus indicators visible
- [ ] Error messages clear and actionable

---

## Browser Compatibility

- Chrome/Edge: 90+
- Firefox: 88+
- Safari: 14+
- Mobile browsers: iOS Safari 14+, Chrome Android 90+

---

## Security Considerations

- ✅ API calls use HTTPS only
- ✅ Input validation on client-side (mirror server validation)
- ✅ No sensitive data stored in localStorage
- ✅ CSRF protection on API endpoints
- ✅ Rate limiting enforced server-side
- ✅ XSS prevention via React's built-in escaping

---

## Success Criteria

### Functional Success

- ✅ All 3 tiers selectable with instant visual feedback
- ✅ Cost estimates show in real-time (<200ms)
- ✅ All API endpoints successfully called
- ✅ Error handling prevents blank screens
- ✅ Mobile responsive (tested on iPhone, Android)

### Business Success

- ✅ Users understand quality tier differences
- ✅ Tier selection influences pricing decision
- ✅ Professional positioning clearly communicated
- ✅ Quality guarantees prominently displayed

### Technical Success

- ✅ Zero TypeScript errors
- ✅ 100% test coverage for new components
- ✅ Lighthouse score >90
- ✅ Bundle size increase <150KB
- ✅ <2s page load time

---

## Known Unknowns & Future Enhancements

### Phase 5.5 (Post-Launch)

- [ ] Video preview capability with selected tier specs
- [ ] User-submitted video validation with QC report display
- [ ] Tier recommendation based on input content
- [ ] Pricing calculator with historical data
- [ ] Comparison slider for before/after quality

### Phase 6 (Future)

- [ ] Custom tier creation for enterprise customers
- [ ] Volume license pricing
- [ ] SLA agreements per tier
- [ ] Quality assurance scoring visualization
- [ ] Performance benchmarking dashboard

---

## Dependencies

### React Libraries Needed

- `react`: 18.2+
- `react-redux`: 8.0+
- `@reduxjs/toolkit`: 1.9+
- `axios`: 1.4+ (for API calls)
- `react-chartjs-2`: For cost breakdown visualization

### Already Installed

- TypeScript
- Tailwind CSS
- Jest

---

## Deployment Plan

### Staging Deployment

1. Deploy to staging environment
2. Run full test suite (unit + E2E)
3. Performance testing
4. Security scanning
5. Stakeholder review

### Production Deployment

1. Feature flag enabled for 10% of users
2. Monitor error rates in Sentry
3. Gradual rollout to 50%, then 100%
4. Keep rollback plan ready (feature flag disable)

---

## Risk Mitigation

### Technical Risks

- **API Timeout**: Add loading states, retry logic
- **Stale Data**: Refresh tier specs on page focus
- **Browser Compatibility**: Polyfills for older browsers
- **Mobile Performance**: Progressive enhancement approach

### Business Risks

- **User Confusion**: Clear tier explanations, comparison table
- **Pricing Concerns**: Show total cost prominently
- **Quality Doubt**: Display certifications and guarantees

---

## Success Metrics Dashboard

Track these KPIs post-launch:

- Tier selection distribution (goal: 30% GOOD, 50% HIGH, 20% ULTRA)
- Average transaction value increase
- User satisfaction with tier choices
- Page load time performance
- Error rate in cost calculation
- Mobile vs desktop usage split

---

## Timeline

**Estimated Duration by Phase**:

- Planning & Design: 4 hours
- Component Implementation: 8 hours
- API Integration: 4 hours
- Testing: 4 hours
- Polish & Optimization: 3 hours
- **Total: 23 hours (~3 business days)**

**Current Status**: Phase 1-4 complete (core framework)
**Next Session**: Begin Phase 5 (React dashboard)

---

## Success Declaration Criteria

Phase 5 is **COMPLETE** when:

1. ✅ All 5 React components created and tested
2. ✅ 3 API endpoints fully integrated
3. ✅ 100% test coverage for new components
4. ✅ Responsive on mobile/tablet/desktop
5. ✅ Lighthouse score >90
6. ✅ WCAG 2.1 AA accessibility compliance
7. ✅ Deployed to production
8. ✅ Documented for future maintenance

---

**Document Created**: February 6, 2026  
**Phase Status**: Ready for implementation  
**Next Action**: Begin React component development
