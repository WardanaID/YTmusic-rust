//! Karaoke vocal cut on everything the app plays, after the equalizer and before
//! crossfeed, mono and balance.
//!
//! Lead vocals sit dead centre in most stereo mixes (`L ≈ R` in both level and phase),
//! while bass and drums share the middle down low, cymbals and air sit up high, and
//! stereo pianos, guitars, keys and room mics either pan off-centre or carry stereo phase
//! width between the ears. An overlapping short-time Fourier transform (`realfft`) looks
//! at each narrow frequency bin in the vocal band (`170 Hz .. 7.5 kHz`) and only subtracts
//! the shared in-phase centre vector in bins whose left and right channels closely match,
//! leaving bass, treble and panned or stereo-miked instruments intact.
//!
//! `Guide` cuts the centre vocal by roughly half (~-6.4 dB) so the singer stays audible
//! to follow; `Full` removes the dry centre vocal altogether. Changes glide over `RAMP_MS`
//! without clicking, and when off and settled frames pass through untouched.

use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// How far the centre vocal is turned down.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VocalCut {
    #[default]
    Off,
    /// Leaves the vocal at roughly half level (~-6.4 dB) to sing along with.
    Guide,
    /// Cuts the dry centre vocal altogether.
    Full,
}

impl VocalCut {
    pub const ALL: [Self; 3] = [Self::Off, Self::Guide, Self::Full];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => crate::tr!("ui-off"),
            Self::Guide => crate::tr!("player-vocal-cut-guide"),
            Self::Full => crate::tr!("player-vocal-cut-full"),
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Guide => "guide",
            Self::Full => "full",
        }
    }

    pub fn from_key(key: &str) -> Self {
        Self::ALL.into_iter().find(|v| v.key() == key).unwrap_or_default()
    }

    fn from_u8(v: u8) -> Self {
        Self::ALL.get(v as usize).copied().unwrap_or_default()
    }

    /// Share of the matched centre vocal removed: ~-6.4 dB (~48% left) on `Guide`, all on `Full`.
    fn amount(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::Guide => 0.52,
            Self::Full => 1.0,
        }
    }
}

/// What the UI wants, read by the device's thread. The default is `Off`.
#[derive(Default)]
pub struct VocalParams(AtomicU8);

impl VocalParams {
    pub fn set(&self, mode: VocalCut) {
        self.0.store(mode as u8, Ordering::Relaxed);
    }
}

/// Vocal frequency taper: untouched below `LOW_START_HZ` and above `HIGH_END_HZ`, full
/// between `LOW_FULL_HZ` and `HIGH_FULL_HZ`, raised-cosine ramps in between.
const LOW_START_HZ: f32 = 170.0;
const LOW_FULL_HZ: f32 = 260.0;
const HIGH_FULL_HZ: f32 = 4200.0;
const HIGH_END_HZ: f32 = 7500.0;

/// Narrow stereo-correlation window (`2 Re(L R*) / (|L|² + |R|²)`): bins below `PAN_START`
/// (panned piano/guitar or wide stereo pads) are left alone; bins above `PAN_FULL` get the
/// full cut.
const PAN_START: f32 = 0.76;
const PAN_FULL: f32 = 0.96;

/// How long a mode change takes: long enough not to click, short enough to feel at once.
const RAMP_MS: f32 = 22.0;

fn band_weight(hz: f32) -> f32 {
    if hz <= LOW_START_HZ || hz >= HIGH_END_HZ {
        0.0
    } else if hz < LOW_FULL_HZ {
        0.5 - 0.5 * (PI * (hz - LOW_START_HZ) / (LOW_FULL_HZ - LOW_START_HZ)).cos()
    } else if hz <= HIGH_FULL_HZ {
        1.0
    } else {
        0.5 + 0.5 * (PI * (hz - HIGH_FULL_HZ) / (HIGH_END_HZ - HIGH_FULL_HZ)).cos()
    }
}

struct Stft {
    n: usize,
    hop: usize,
    at: usize,
    win: Vec<f32>,
    freq_weight: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    hist_l: Vec<f32>,
    hist_r: Vec<f32>,
    ola_l: Vec<f32>,
    ola_r: Vec<f32>,
    time_buf: Vec<f32>,
    spec_l: Vec<Complex<f32>>,
    spec_r: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    ifft_scratch: Vec<Complex<f32>>,
    raw_mask: Vec<f32>,
    smooth_mask: Vec<f32>,
}

impl Stft {
    fn new(rate: u32) -> Self {
        let rate = rate.max(8_000);
        // ~21 ms window (1024 at 48 kHz) with 4× overlap (75%): fine enough (~47 Hz bins)
        // to separate bass and panned notes from vocal harmonics without noticeable lag.
        let n = ((rate as usize * 1024) / 48_000).next_power_of_two().clamp(256, 4096);
        let hop = n / 4;
        let bins = n / 2 + 1;
        // Periodic sqrt-Hann: `win²` across four quarter-shifted hops sums to 2 everywhere.
        let win = (0..n)
            .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()).sqrt() as f32)
            .collect();
        let nyquist_cap = rate as f32 * 0.45;
        let mut freq_weight: Vec<f32> = (0..bins)
            .map(|k| {
                let hz = k as f32 * rate as f32 / n as f32;
                if hz >= nyquist_cap { 0.0 } else { band_weight(hz) }
            })
            .collect();
        freq_weight[0] = 0.0;
        freq_weight[bins - 1] = 0.0;

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);
        let time_buf = fft.make_input_vec();
        let spec_l = fft.make_output_vec();
        let spec_r = fft.make_output_vec();
        let fft_scratch = fft.make_scratch_vec();
        let ifft_scratch = ifft.make_scratch_vec();
        Self {
            n,
            hop,
            at: 0,
            win,
            freq_weight,
            fft,
            ifft,
            hist_l: vec![0.0; n],
            hist_r: vec![0.0; n],
            ola_l: vec![0.0; n],
            ola_r: vec![0.0; n],
            time_buf,
            spec_l,
            spec_r,
            fft_scratch,
            ifft_scratch,
            raw_mask: vec![0.0; bins],
            smooth_mask: vec![0.0; bins],
        }
    }

    fn reset(&mut self) {
        self.at = 0;
        self.hist_l.fill(0.0);
        self.hist_r.fill(0.0);
        self.ola_l.fill(0.0);
        self.ola_r.fill(0.0);
        self.raw_mask.fill(0.0);
        self.smooth_mask.fill(0.0);
    }

    fn step(&mut self, x: [f32; 2], cut: f32) -> [f32; 2] {
        let wet = [self.ola_l[self.at], self.ola_r[self.at]];
        let write = self.n - self.hop + self.at;
        self.hist_l[write] = x[0];
        self.hist_r[write] = x[1];
        self.at += 1;
        if self.at == self.hop {
            self.ola_l.copy_within(self.hop..self.n, 0);
            self.ola_l[self.n - self.hop..self.n].fill(0.0);
            self.ola_r.copy_within(self.hop..self.n, 0);
            self.ola_r[self.n - self.hop..self.n].fill(0.0);
            self.run_hop(cut);
            self.hist_l.copy_within(self.hop..self.n, 0);
            self.hist_r.copy_within(self.hop..self.n, 0);
            self.at = 0;
        }
        wet
    }

    fn run_hop(&mut self, cut: f32) {
        let Self {
            n,
            win,
            freq_weight,
            fft,
            ifft,
            hist_l,
            hist_r,
            ola_l,
            ola_r,
            time_buf,
            spec_l,
            spec_r,
            fft_scratch,
            ifft_scratch,
            raw_mask,
            smooth_mask,
            ..
        } = self;
        let bins = raw_mask.len();

        for (dst, (&s, &w)) in time_buf.iter_mut().zip(hist_l.iter().zip(win.iter())) {
            *dst = s * w;
        }
        let _ = fft.process_with_scratch(time_buf, spec_l, fft_scratch);

        for (dst, (&s, &w)) in time_buf.iter_mut().zip(hist_r.iter().zip(win.iter())) {
            *dst = s * w;
        }
        let _ = fft.process_with_scratch(time_buf, spec_r, fft_scratch);

        raw_mask[0] = 0.0;
        raw_mask[bins - 1] = 0.0;
        for (((dst, &wf), &l), &r) in raw_mask[1..bins - 1]
            .iter_mut()
            .zip(&freq_weight[1..bins - 1])
            .zip(&spec_l[1..bins - 1])
            .zip(&spec_r[1..bins - 1])
        {
            if wf == 0.0 {
                *dst = 0.0;
                continue;
            }
            let el = l.norm_sqr();
            let er = r.norm_sqr();
            let etot = el + er;
            if etot <= 1e-18 {
                *dst = 0.0;
                continue;
            }
            let corr = (2.0 * (l.re * r.re + l.im * r.im) / etot).clamp(-1.0, 1.0);
            let u = ((corr - PAN_START) / (PAN_FULL - PAN_START)).clamp(0.0, 1.0);
            *dst = wf * u * u * (3.0 - 2.0 * u);
        }

        for ((((win3, sm), &wf), l), r) in raw_mask
            .windows(3)
            .zip(&mut smooth_mask[1..bins - 1])
            .zip(&freq_weight[1..bins - 1])
            .zip(&mut spec_l[1..bins - 1])
            .zip(&mut spec_r[1..bins - 1])
        {
            if wf == 0.0 {
                *sm = 0.0;
                continue;
            }
            let local = 0.25 * win3[0] + 0.5 * win3[1] + 0.25 * win3[2];
            let next = *sm + 0.55 * (local - *sm);
            *sm = if next < 1e-6 { 0.0 } else { next };
            let remove = cut * *sm;
            if remove <= 0.0 {
                continue;
            }
            // Cap the subtracted vector at the quieter ear's magnitude so an off-centre
            // instrument never gets over-subtracted into anti-phase on the far side.
            let mid = (*l + *r) * 0.5;
            let mid_abs = mid.norm();
            let cap = l.norm_sqr().min(r.norm_sqr()).sqrt();
            let center = if mid_abs > cap && mid_abs > 1e-12 { mid * (cap / mid_abs) } else { mid };
            *l -= center * remove;
            *r -= center * remove;
        }
        // `ComplexToReal` requires purely real DC and Nyquist bins.
        spec_l[0].im = 0.0;
        spec_l[bins - 1].im = 0.0;
        spec_r[0].im = 0.0;
        spec_r[bins - 1].im = 0.0;

        let scale = 1.0 / (2.0 * *n as f32);
        let flush = |v: f32| if v.abs() < 1e-20 { 0.0 } else { v };

        let _ = ifft.process_with_scratch(spec_l, time_buf, ifft_scratch);
        for (dst, (&s, &w)) in ola_l.iter_mut().zip(time_buf.iter().zip(win.iter())) {
            *dst = flush(*dst + s * w * scale);
        }

        let _ = ifft.process_with_scratch(spec_r, time_buf, ifft_scratch);
        for (dst, (&s, &w)) in ola_r.iter_mut().zip(time_buf.iter().zip(win.iter())) {
            *dst = flush(*dst + s * w * scale);
        }
    }
}

pub struct Vocal {
    params: Arc<VocalParams>,
    step: f32,
    raw: u8,
    want: VocalCut,
    stft: Stft,
    /// Wet/dry blend `0..=1` (off vs active) and vocal-band cut `0..=1` (`Guide` vs `Full`).
    mix: f32,
    cut: f32,
    /// Off and settled: frames pass through untouched.
    idle: bool,
}

impl Vocal {
    pub fn new(params: Arc<VocalParams>, rate: u32) -> Self {
        let rate = rate.max(8_000);
        let raw = params.0.load(Ordering::Relaxed);
        let want = VocalCut::from_u8(raw);
        let (mix, cut, idle) = match want {
            VocalCut::Off => (0.0, 1.0, true),
            mode => (1.0, mode.amount(), false),
        };
        Self {
            params,
            step: 1.0 / (rate as f32 * RAMP_MS / 1000.0).max(1.0),
            raw,
            want,
            stft: Stft::new(rate),
            mix,
            cut,
            idle,
        }
    }

    fn read(&mut self) {
        self.raw = self.params.0.load(Ordering::Relaxed);
        self.want = VocalCut::from_u8(self.raw);
        if self.mix == 0.0 && self.want != VocalCut::Off {
            self.cut = self.want.amount();
        }
        self.idle = false;
    }

    /// Spectral masking can reshape transient peaks slightly; keep the limiter active.
    pub fn boosts(&self) -> bool {
        !self.idle
    }

    fn glide(&mut self) {
        let toward = |v: &mut f32, to: f32, by: f32| *v = if (to - *v).abs() <= by { to } else { *v + by.copysign(to - *v) };
        let want_mix = if self.want == VocalCut::Off { 0.0 } else { 1.0 };
        toward(&mut self.mix, want_mix, self.step);
        if self.want != VocalCut::Off {
            toward(&mut self.cut, self.want.amount(), self.step);
        }
        if self.mix == 0.0 && self.want == VocalCut::Off {
            self.stft.reset();
            self.idle = true;
        }
    }

    /// One frame, its channels in the device's order (front left and right first).
    pub fn process(&mut self, frame: &mut [f32]) {
        if self.params.0.load(Ordering::Relaxed) != self.raw {
            self.read();
        }
        if self.idle {
            return;
        }
        let [l, r, ..] = frame else { return };
        if !l.is_finite() || !r.is_finite() {
            self.stft.reset();
            return;
        }
        self.glide();
        if self.idle {
            return;
        }
        let x = [*l, *r];
        let wet = self.stft.step(x, self.cut);
        if !wet.iter().all(|s| s.is_finite()) {
            self.stft.reset();
            return;
        }
        (*l, *r) = (x[0] + self.mix * (wet[0] - x[0]), x[1] + self.mix * (wet[1] - x[1]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocal(mode: VocalCut) -> (Vocal, Arc<VocalParams>) {
        let p = Arc::new(VocalParams::default());
        p.set(mode);
        (Vocal::new(p.clone(), 48_000), p)
    }

    fn run(v: &mut Vocal, frame: [f32; 2]) -> [f32; 2] {
        let mut f = frame;
        v.process(&mut f);
        f
    }

    /// Steady-state peak of each side for a sine at `f` Hz, `amp` on each input side.
    fn peaks(v: &mut Vocal, f: f64, amp: [f32; 2]) -> [f32; 2] {
        let mut peak = [0f32; 2];
        for k in 0..48_000 {
            let s = (2.0 * std::f64::consts::PI * f * k as f64 / 48_000.0).sin() as f32;
            let y = run(v, [s * amp[0], s * amp[1]]);
            if k > 24_000 {
                peak = std::array::from_fn(|i| peak[i].max(y[i].abs()));
            }
        }
        peak
    }

    fn noise() -> impl Iterator<Item = [f32; 2]> {
        let mut seed = 1u32;
        std::iter::repeat_with(move || {
            std::array::from_fn(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1 << 23) as f32 - 1.0
            })
        })
    }

    #[test]
    fn modes_round_trip() {
        for m in VocalCut::ALL {
            assert_eq!(VocalCut::from_key(m.key()), m);
            assert_eq!(VocalCut::from_u8(m as u8), m);
        }
        assert_eq!(VocalCut::from_key("unknown"), VocalCut::Off);
        assert_eq!(VocalCut::from_u8(99), VocalCut::Off);
    }

    #[test]
    fn off_passes_samples_through_unchanged() {
        let (mut v, _) = vocal(VocalCut::Off);
        for x in noise().take(10_000).chain([[f32::MAX, f32::MIN_POSITIVE], [-0.0, f32::NAN]]) {
            let y = run(&mut v, x);
            assert_eq!(y.map(f32::to_bits), x.map(f32::to_bits));
        }
        assert!(v.idle && !v.boosts());
        // Surround channels past front left/right and mono frames are never touched.
        let (mut v, _) = vocal(VocalCut::Full);
        let mut surround = [0.4, 0.4, 0.3, 0.5, 0.6, 0.7];
        v.process(&mut surround);
        assert_eq!(surround[2..], [0.3, 0.5, 0.6, 0.7]);
        let mut mono = [0.7];
        v.process(&mut mono);
        assert_eq!(mono, [0.7]);
    }

    #[test]
    fn cuts_centre_voices_and_preserves_bass_treble_and_panned_instruments() {
        let (mut full, _) = vocal(VocalCut::Full);
        // A 1 kHz voice dead centre is removed (> 50 dB down).
        let voice = peaks(&mut full, 1_000.0, [0.5, 0.5]);
        assert!(voice.iter().all(|&p| p < 0.002), "full voice: {voice:?}");

        // Bass (60 Hz) and air (12 kHz) in the middle stay whole.
        let (mut full, _) = vocal(VocalCut::Full);
        let bass = peaks(&mut full, 60.0, [0.5, 0.5]);
        assert!(bass.iter().all(|&p| (p - 0.5).abs() < 0.02), "bass: {bass:?}");
        let (mut full, _) = vocal(VocalCut::Full);
        let air = peaks(&mut full, 12_000.0, [0.5, 0.5]);
        assert!(air.iter().all(|&p| (p - 0.5).abs() < 0.02), "air: {air:?}");

        // A piano or guitar panned off-centre (L=0.5, R=0.2) right in the vocal range
        // (1 kHz) keeps its full stereo balance without phase inversion.
        let (mut full, _) = vocal(VocalCut::Full);
        let [l_pan, r_pan] = peaks(&mut full, 1_000.0, [0.5, 0.2]);
        assert!((l_pan - 0.5).abs() < 0.02 && (r_pan - 0.2).abs() < 0.02, "panned: {l_pan} {r_pan}");

        // Guide mode leaves roughly half the vocal (~48% of 0.5 = 0.24) to follow,
        // clearly distinct from both Off (0.50) and Full (0.00).
        let (mut guide, _) = vocal(VocalCut::Guide);
        let guide_voice = peaks(&mut guide, 1_000.0, [0.5, 0.5]);
        assert!(guide_voice.iter().all(|&p| (p - 0.24).abs() < 0.03), "guide: {guide_voice:?}");
    }

    #[test]
    fn turning_off_returns_to_untouched_passthrough() {
        let (mut v, p) = vocal(VocalCut::Full);
        noise().take(3_000).for_each(|x| _ = run(&mut v, x));
        p.set(VocalCut::Off);
        noise().take(3_000).for_each(|x| _ = run(&mut v, x));
        assert!(v.idle && !v.boosts());
        for x in noise().take(1_000) {
            assert_eq!(run(&mut v, x), x);
        }
    }

    #[test]
    fn changes_glide_without_clicking() {
        for from in VocalCut::ALL {
            for to in VocalCut::ALL {
                let (mut v, p) = vocal(from);
                let w = 2.0 * std::f64::consts::PI * 440.0 / 48_000.0;
                let tone = |k: usize| (w * k as f64).sin() as f32 * 0.5;
                (0..4_800).for_each(|k| _ = run(&mut v, [tone(k), tone(k)]));
                p.set(to);
                let mut last = run(&mut v, [tone(4_800), tone(4_800)]);
                for k in 4_801..9_600 {
                    let y = run(&mut v, [tone(k), tone(k)]);
                    assert!(y.iter().zip(last).all(|(a, b)| (a - b).abs() < 0.045), "{from:?} to {to:?}: {last:?} {y:?} at {k}");
                    last = y;
                }
                assert_eq!(v.boosts(), to != VocalCut::Off, "{from:?} to {to:?}");
            }
        }
    }

    #[test]
    fn a_broken_sample_doesnt_stick_and_silence_flushes_denormals() {
        let (mut v, _) = vocal(VocalCut::Full);
        assert_eq!(run(&mut v, [f32::NAN, 0.25])[1], 0.25);
        assert!(noise().take(2_000).all(|x| run(&mut v, x).iter().all(|s| s.is_finite())));
        (0..8_000).for_each(|_| _ = run(&mut v, [0.0, 0.0]));
        assert!(v.stft.ola_l.iter().chain(&v.stft.ola_r).all(|&x| x == 0.0 || x.is_normal()));
    }
}
