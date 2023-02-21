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


def m4(num: int):
    16 if num < 16 else mod4(num)

def fft3d(clip: vs.VideoNode, **kwargs):
    if hasattr(core, 'neo_fft3d'):
      return core.neo_fft3d.FFT3D(clip, **kwargs)
    else:
      return core.fft3dfilter.FFT3DFilter(clip, **kwargs)

@dataclass
class TemporalDegrain2:
    degrainTR: int = 1
    # TODO: change to planes? Or support list of planes?
    degrainPlane: int = 4
    grainLevel: int = 2
    grainLevelSetup: bool = False
    meAlg: int = 4
    meAlgPar: int | None = None
    meSubpel: int | None = None
    meBlksz: int | None = None
    meTM: bool = False
    limitSigma: int | None = None
    limitBlksz: int | None = None
    fftThreads: int | None = None
    postFFT: int = 0
    postTR: int = 1
    postSigma: int = 1
    postMix: int = 0
    postBlkSize: int | None = None
    knlDevId: int | None = 0
    ppSAD1: int | None = None
    ppSAD2: int | None = None
    ppSCD1: int | None = None
    thSCD2: int = 50
    DCT: int = 0
    SubPelInterp: int = 2
    SrchClipPP: int | Prefilter | None =  None
    GlobalMotion: bool = True
    ChromaMotion: bool = True
    rec: bool = False # TODO:  rename 'refine'?
    extraSharp: bool = False
    outputStage: int = 2
    limiter: Callable[[vs.VideoNode], vs.VideoNode] | None = None # Function used to limit the maximum denoising effect MVDegrain can have. Defaults to custom FFT3DFilter

    def denoise(self, clip: vs.VideoNode) -> vs.VideoNode:
        # TODO allow parameter overrides...
        width = clip.width
        height = clip.height

        neutral = get_neutral_value(clip)
        isFLOAT = get_sample_type(clip) == vs.FLOAT
        isGRAY = get_color_family(clip) == vs.GRAY

        degrainPlane = self.degrainPlane
        ChromaNoise = degrainPlane > 0
        ChromaMotion = self.ChromaMotion
        if isGRAY:
            ChromaMotion = False
            ChromaNoise = False
            degrainPlane = 0

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

        fPlane = [[0], [1], [2], [1,2], [0,1,2]][degrainPlane]

        postFFT = self.postFFT
        postTR = self.postTR
        postSigma = self.postSigma
        postBlkSize = fallback(self.postBlkSize, [0,48,32,12,0,0][self.postFFT])
        postMix = self.postMix
        if postFFT <= 0:
            postTR = 0
        if postFFT == 3:
            postTR = min(postTR, 7)
        if postFFT in [1, 2]:
            postTR = min(postTR, 2)

        knlDevId = self.knlDevId
        fftThreads = self.fftThreads

        refine = 1 if self.rec else 0
        meSubpel = fallback(self.meSubpel, [4, 2, 2, 1][autoTune])
        meBlksz = fallback(self.meBlksz, [8, 8, 16, 32][autoTune])
        meSharp = self.SubPelInterp
        meAlg = self.meAlg
        meTM = self.meTM
        Overlap = meBlksz // 2
        Lambda = (1000 if self.meTM else 100) * (meBlksz ** 2) // 64
        PNew = 50 if self.meTM else 25
        GlobalMotion = self.GlobalMotion
        DCT = self.DCT

        # radius/range parameter for the motion estimation algorithms
        # AVS version uses the following, but values seemed to be based on an
        # incorrect understanding of the MVTools motion seach algorithm, mistaking 
        # it for the exact x264 behavior.
        # meAlgPar = [2,2,2,2,16,24,2,2][meAlg] 
        # Using Dogway's SMDegrain options here instead of the TemporalDegrain2 AVSI versions, which seem wrong.
        meAlgPar = fallback(self.meAlgPar, 5 if self.rec and self.meTM else 2)

        limitAT = [-1, -1, 0, 0, 0, 1][grainLevel] + autoTune + 1
        limitSigma = fallback(self.limitSigma, [6,8,12,16,32,48][limitAT])
        limitBlksz = fallback(self.limitBlksz, [12,16,24,32,64,96][limitAT])

        SrchClipPP = fallback(self.SrchClipPP, [0,0,0,3,3,3][self.grainLevel])
        CMplanes = [0,1,2] if self.ChromaMotion else [0]
        rad = 3 if self.extraSharp else None
        hpad = meBlksz
        vpad = meBlksz
        postTD  = postTR * 2 + 1
        maxTR = max(degrainTR, postTR)

        ppSAD1 = fallback(self.ppSAD1,[3,5,7,9,11,13][grainLevel])
        ppSAD2 = fallback(self.ppSAD2,[2,4,5,6,7,8][grainLevel])
        ppSCD1 = fallback(self.ppSCD1,[3,3,3,4,5,6][grainLevel])

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
                      planes=fPlane)

        # Run motion analysis on the widest tr that we'll use for any operation,
        # whether degrain or post, and then reuse them for all following operations.
        maxMV = MVTools(clip, **preset(tr=maxTR))
        maxMV.analyze()

        # First MV-denoising stage. Usually here's some temporal-medianfiltering going on.
        # For simplicity, we just use MDegrain.
        mv1 = MVTools(clip, vectors=maxMV, **preset)
        NR1 = mv1.degrain(thSAD=thSAD1, thSCD=(thSCD1, thSCD2))

        if degrainTR > 0:
            spatD = limiter(clip)

            # Limit NR1 to not do more than what "spat" would do.
            NR1D = core.std.MakeDiff(clip, NR1)
            expr = 'x abs y abs < x y ?' if isFLOAT else f'x {neutral} - abs y {neutral} - abs < x y ?'
            DD   = core.std.Expr([spatD, NR1D], [expr])
            NR1x = core.std.MakeDiff(clip, DD, [0])

            # Second MV-denoising stage. We use MDegrain.
            mv2 = MVTools(NR1x, vectors=maxMV, **preset)
            NR2 = mv2.degrain(thSAD=thSAD2, thSCD=(thSCD1, thSCD2))
        else:
            NR2 = clip

        postDenoiser = [
                partial(removegrain, mode=1),
                partial(fft3d, sigma=postSigma, planes=fPlane, bt=postTD, ncpu=fftThreads, bw=postBlkSize, bh=postBlkSize),
                partial(fft3d, sigma=postSigma, planes=fPlane, bt=postTD, ncpu=fftThreads, bw=postBlkSize, bh=postBlkSize),
                partial(DFTTest.denoise, sigma=postSigma * 4, tbsize=postTD, planes=fPlane, sbsize=postBlkSize, sosize=postBlkSize * 9/12),
                partial(nl_means, strength=postSigma / 2, tr=postTR, sr=2, device_id=knlDevId, planes=fPlane),
                ][postFFT]

        if postTR > 0:
            mvNoiseWindow = MVTools(NR2, vectors=mv1, **preset(tr=postTR))
            dnWindow = mvNoiseWindow.compensate(postDenoiser, thSAD=thSAD2, )
        else:
            dnWindow = postDenoiser(NR2)

        sharpened = contrasharpening(dnWindow, clip, rad)

        if postMix > 0:
            sharpened = core.std.Expr([clip,sharpened],f"x {postMix} * y {100-postMix} * + 100 /")

        return [NR1x, NR2, sharpened][outputStage]


