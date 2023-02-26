import vapoursynth as vs
from typing import Callable
from functools import partial
from dataclasses import dataclass
from vstools import fallback, get_color_family, get_sample_type, mod4, core, get_neutral_value
from vsrgtools import contrasharpening, removegrain
from vsdenoise.dfttest import DFTTest
from vsdenoise.knlm import nl_means

from vsdenoise.mvtools import MVTools, MVToolsPresets, SADMode, SearchMode, MotionMode
from vsdenoise.prefilters import Prefilter


__all__ = [
        'TemporalDegrain2'
        ]

## differences
# degrainPlane => planes
# rec => refine
# Refine can now be an int to describe the number of times/steps to refine

# thSCD2 range is 0-100, per the vs-denoise standard

# Support for passing in custom prefilters functions, or prefiltered clips
# Support for passing in custom post denoisers
@dataclass
class TemporalDegrain2:
    # Main tunables
    degrainTR: int = 1
    grainLevel: int = 2

    # TODO: change to planes? Or support list of planes?
    # Add support for array or int and call it 'planes'
    planes: int | list[int]  = 4

    # TODO: Support custom functions
    postFFT: int | Callable[[vs.VideoNode], vs.VideoNode] = 0
    postSigma: int = 1

    # Tuning / output params
    grainLevelSetup: bool = False
    outputStage: int = 2

#  Motion params
    meAlg: int = 4
    meAlgPar: int | None = None
    meSubpel: int | None = None
    meBlksz: int | None = None
    meTM: bool = False
    ppSAD1: int | None = None
    ppSAD2: int | None = None
    ppSCD1: int | None = None
    thSCD2: int = 50
    DCT: int = 0
    SubPelInterp: int = 2
    SrchClipPP: int | Prefilter | None =  None
    GlobalMotion: bool = True
    ChromaMotion: bool = True
    refine: bool | int = False 

# denoisers
    limiter: Callable[[vs.VideoNode], vs.VideoNode] | None = None # Function used to limit the maximum denoising effect MVDegrain can have. Defaults to custom FFT3DFilter
    limitSigma: int | None = None
    limitBlksz: int | None = None
    knlDevId: int | None = 0

# post denoising
    postTR: int = 1
    postMix: int = 0
    postBlkSize: int | None = None

# various knobs
# TODO support ints allow choosing a custom sharpening radius.
    extraSharp: bool = False
    fftThreads: int | None = None

    def denoise(self, clip: vs.VideoNode) -> vs.VideoNode:
        # TODO allow parameter overrides...
        width = clip.width
        height = clip.height

        neutral = get_neutral_value(clip)
        isFLOAT = get_sample_type(clip) == vs.FLOAT
        isGRAY = get_color_family(clip) == vs.GRAY

        planes = self.planes
        ChromaMotion = self.ChromaMotion

        if isGRAY:
            ChromaMotion = False
            planes = 0

        longlat = max(width, height)
        shortlat = min(width, height)
        # Scale grainLevel from -2-3 -> 0-5
        grainLevel = self.grainLevel + 2

        degrainTR = self.degrainTR
        outputStage = self.outputStage
        if self.grainLevelSetup:
            outputStage = 0
            degrainTR = 3

        if (longlat<=1050 and shortlat<=576):
            autoTune = 0
        elif (longlat<=1280 and shortlat<=720):
            autoTune = 1
        elif (longlat<=2048 and shortlat<=1152):
            autoTune = 2
        else:
            autoTune = 3

        if isinstance(planes, int):
            # Convert int-based plane selection to array-based plane selection to match the normal VS standard
            planes = [[0], [1], [2], [1,2], [0,1,2]][planes]

        postTR = self.postTR
        postTD  = postTR * 2 + 1
        postSigma = self.postSigma
        postMix = self.postMix
        knlDevId = self.knlDevId
        fftThreads = self.fftThreads

        postFFT = self.postFFT
        if isinstance(postFFT, int):
            postBlkSize = fallback(self.postBlkSize, [0,48,32,12,0,0][postFFT])
            if postFFT <= 0:
                postTR = 0
            if postFFT == 3:
                postTR = min(postTR, 7)
            if postFFT in [1, 2]:
                postTR = min(postTR, 2)

            postDenoiser = [
                    partial(removegrain, mode=1),
                    partial(fft3d, sigma=postSigma, planes=planes, bt=postTD, ncpu=fftThreads, bw=postBlkSize, bh=postBlkSize),
                    partial(fft3d, sigma=postSigma, planes=planes, bt=postTD, ncpu=fftThreads, bw=postBlkSize, bh=postBlkSize),
                    partial(DFTTest.denoise, sigma=postSigma * 4, tbsize=postTD, planes=planes, sbsize=postBlkSize, sosize=postBlkSize * 9/12),
                    partial(nl_means, strength=postSigma / 2, tr=postTR, sr=2, device_id=knlDevId, planes=planes),
                    ][postFFT]
        else:
            postDenoiser = postFFT


        SrchClipPP = fallback(self.SrchClipPP, [0,0,0,3,3,3][self.grainLevel])

        maxTR = max(degrainTR, postTR)

        if isinstance(self.refine, bool):
            refine = 1 if self.refine else 0
        else:
            refine = self.refine

        meTM = self.meTM

        # radius/range parameter for the motion estimation algorithms
        # AVS version uses the following, but values seemed to be based on an
        # incorrect understanding of the MVTools motion seach algorithm, mistaking 
        # it for the exact x264 behavior.
        # meAlgPar = [2,2,2,2,16,24,2,2][meAlg] 
        # Using Dogway's SMDegrain options here instead of the TemporalDegrain2 AVSI versions, which seem wrong.
        meAlgPar = fallback(self.meAlgPar, 5 if refine and meTM else 2)
        meAlg = self.meAlg

        meSubpel = fallback(self.meSubpel, [4, 2, 2, 1][autoTune])
        meBlksz = fallback(self.meBlksz, [8, 8, 16, 32][autoTune])
        hpad = meBlksz
        vpad = meBlksz
        meSharp = self.SubPelInterp
        Overlap = meBlksz // 2
        Lambda = (1000 if meTM else 100) * (meBlksz ** 2) // 64
        PNew = 50 if meTM else 25
        GlobalMotion = self.GlobalMotion
        DCT = self.DCT
        ppSAD1 = fallback(self.ppSAD1,[3,5,7,9,11,13][grainLevel])
        ppSAD2 = fallback(self.ppSAD2,[2,4,5,6,7,8][grainLevel])
        ppSCD1 = fallback(self.ppSCD1,[3,3,3,4,5,6][grainLevel])
        CMplanes = [0,1,2] if ChromaMotion else [0]

        if DCT == 5:
            #rescale threshold to match the SAD values when using SATD
            ppSAD1 *= 1.7
            ppSAD2 *= 1.7
            # ppSCD1 - this must not be scaled since scd is always based on SAD independently of the actual dct setting

        #here the per-pixel measure is converted to the per-8x8-Block (8*8 = 64) measure MVTools is using
        thSAD1 = int(ppSAD1 * 64)
        thSAD2 = int(ppSAD2 * 64)
        thSCD1 = int(ppSCD1 * 64)
        thSCD2 = self.thSCD2

        limitAT = [-1, -1, 0, 0, 0, 1][grainLevel] + autoTune + 1
        limitSigma = fallback(self.limitSigma, [6,8,12,16,32,48][limitAT])
        limitBlksz = fallback(self.limitBlksz, [12,16,24,32,64,96][limitAT])

        sharpenRadius = 3 if self.extraSharp else None

        def limiterFFT3D(clip: vs.VideoNode) -> vs.VideoNode:
            s2 = limitSigma * 0.625
            s3 = limitSigma * 0.375
            s4 = limitSigma * 0.250
            ovNum = [4, 4, 4, 3, 2, 2][grainLevel]
            ov = 2 * round(limitBlksz / ovNum * 0.5)

            spat = fft3d(clip, planes=CMplanes, sigma=limitSigma, sigma2=s2, sigma3=s3, sigma4=s4, bt=3, bw=limitBlksz, bh=limitBlksz, ow=ov, oh=ov, ncpu=fftThreads)
            return core.std.MakeDiff(clip, spat)

        limiter = fallback(self.limiter, limiterFFT3D)

        # Blur image and soften edges to assist in motion matching of edge blocks. Blocks are matched by SAD (sum of absolute differences between blocks), but even
        # a slight change in an edge from frame to frame will give a high SAD due to the higher contrast of edges
        if isinstance(SrchClipPP, Prefilter):
            srchClip = SrchClipPP(clip)
        else:
            srchClip = [Prefilter.NONE, Prefilter.SCALEDBLUR, Prefilter.GAUSSBLUR1, Prefilter.GAUSSBLUR2][SrchClipPP](clip)


        # TODO Add thSADC support, like AVS version
        preset = MVToolsPresets.CUSTOM(tr=degrainTR, refine=refine, prefilter=srchClip,
                      pel=meSubpel, hpad=hpad, vpad=vpad, sharp=meSharp,
                      block_size=meBlksz, overlap=Overlap, 
                      search=SearchMode(meAlg)(recalc_mode=SearchMode(meAlg), param=meAlgPar, pel=meSubpel),
                      motion=MotionMode.MANUAL(truemotion=meTM, coherence=Lambda, pnew=PNew, pglobal=GlobalMotion),
                      sad_mode=(SADMode(DCT), SADMode(DCT)),
                      super_args=dict(chroma=ChromaMotion),
                      analyze_args=dict(chroma=ChromaMotion),
                      recalculate_args=dict(thsad=thSAD1 // 2, lambda_=Lambda//4),
                      planes=planes)

        # Run motion analysis on the widest tr that we'll use for any operation,
        # whether degrain or post, and then reuse them for all following operations.
        maxMV = MVTools(clip, **preset(tr=maxTR))
        maxMV.analyze()

        # First MV-denoising stage. Usually here's some temporal-medianfiltering going on.
        # For simplicity, we just use MDegrain.
        NR1 = MVTools(clip, vectors=maxMV, **preset).degrain(thSAD=thSAD1, thSCD=(thSCD1, thSCD2))

        if degrainTR > 0:
            spatD = limiter(clip)

            # Limit NR1 to not do more than what "spat" would do.
            NR1D = core.std.MakeDiff(clip, NR1)
            expr = 'x abs y abs < x y ?' if isFLOAT else f'x {neutral} - abs y {neutral} - abs < x y ?'
            DD   = core.std.Expr([spatD, NR1D], [expr])
            NR1x = core.std.MakeDiff(clip, DD, [0])

            # Second MV-denoising stage. We use MDegrain.
            NR2 = MVTools(NR1x, vectors=maxMV, **preset).degrain(thSAD=thSAD2, thSCD=(thSCD1, thSCD2))
        else:
            NR2 = clip

        # Post (final stage) denoising.
        if postTR > 0:
            mvNoiseWindow = MVTools(NR2, vectors=maxMV, **preset(tr=postTR))
            dnWindow = mvNoiseWindow.compensate(postDenoiser, thSAD=thSAD2, thSCD=(thSCD1, thSCD2) )
        else:
            dnWindow = postDenoiser(NR2)

        sharpened = contrasharpening(dnWindow, clip, sharpenRadius)

        if postMix > 0:
            sharpened = core.std.Expr([clip,sharpened],f"x {postMix} * y {100-postMix} * + 100 /")

        return [NR1x, NR2, sharpened][outputStage]


def m4(num: int):
    16 if num < 16 else mod4(num)

def fft3d(clip: vs.VideoNode, **kwargs):
    if hasattr(core, 'neo_fft3d'):
      return core.neo_fft3d.FFT3D(clip, **kwargs)
    else:
      return core.fft3dfilter.FFT3DFilter(clip, **kwargs)
