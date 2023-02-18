import vapoursynth as vs
from dataclasses import dataclass
from vstools import fallback, get_color_family, get_depth, get_sample_type, mod4, core, get_neutral_value
from vsrgtools import removegrain, gauss_blur

from vsdenoise.mvtools import MVTools 

def m4(num: int):
    16 if num < 16 else mod4(num)

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
    thSCD2: int = 128
    DCT: int = 0
    SubPelInterp: int = 2
    SrchClipPP: int | None =  None
    GlobalMotion: bool = True
    ChromaMotion: bool = True
    rec: bool = False
    extraSharp: bool = False
    outputStage: int = 2

    def denoise(self, clip: vs.VideoNode) -> vs.VideoNode:
        # TODO allow parameter overrides...
        # TODO use fallback to specifiy tuned defaults.
        width = clip.width
        height = clip.height

        bd = get_depth(clip)
        neutral = get_neutral_value(clip)
        isFLOAT = get_sample_type(clip) == vs.FLOAT
        isGRAY = get_color_family(clip) == vs.GRAY
        # Seems to be used for some kind of bit depth scaling...
        bitDepthMultiplier = 0.00392 if isFLOAT else 1 << (bd - 8)

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
        ouputStage = self.outputStage

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

        if degrainPlane == 0:
            fPlane = [0]
        elif degrainPlane == 1:
            fPlane = [1]
        elif degrainPlane == 2:
            fPlane = [2]
        elif degrainPlane == 3:
            fPlane = [1, 2]
        else:
            fPlane = [0, 1, 2]

        postTR = self.postTR
        if self.postFFT <= 0:
            postTR = 0
        if self.postFFT == 3:
            postTR = min(self.postTR, 7)
        if self.postFFT in [1, 2]:
            postTR = min(self.postTR, 2)

        postBlkSize = fallback(self.postBlkSize, [0,48,32,12,0,0][self.postFFT])

        meSubpel = fallback(self.meSubpel, [4, 2, 2, 1][autoTune])
        meBlksz = fallback(self.meBlksz, [8, 8, 16, 32][autoTune])

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
        Overlap = meBlksz / 2
        Lambda = (1000 if self.meTM else 100) * (meBlksz ** 2) // 64
        PNew = 50 if self.meTM else 25

        ppSAD1 = fallback(self.ppSAD1,[3,5,7,9,11,13][grainLevel])
        ppSAD2 = fallback(self.ppSAD2,[2,4,5,6,7,8][grainLevel])
        ppSCD1 = fallback(self.ppSCD1,[3,3,3,4,5,6][grainLevel])

        if self.DCT == 5:
            #rescale threshold to match the SAD values when using SATD
            ppSAD1 *= 1.7
            ppSAD2 *= 1.7
            # ppSCD1 - this must not be scaled since scd is always based on SAD independently of the actual dct setting

        #here the per-pixel measure is converted to the per-8x8-Block (8*8 = 64) measure MVTools is using
        thSAD1 = int(ppSAD1 * 64)
        thSAD2 = int(ppSAD2 * 64)
        thSCD1 = int(ppSCD1 * 64)
        thSCD2 = self.thSCD2

        # TODO: This needs work.
        super_args = dict(pel=self.meSubpel)

        # TODO: Compare this with QTGMC's implementation.
        # TODO: vs-denoise has prefilters for most of these.
        # Blur image and soften edges to assist in motion matching of edge blocks. Blocks are matched by SAD (sum of absolute differences between blocks), but even
        # a slight change in an edge from frame to frame will give a high SAD due to the higher contrast of edges
        if SrchClipPP == 1:
            # TODO Check to see if rgmode 12 works, if not just use 11 as its identical.
            spatialBlur = removegrain(clip.resize.Bilinear(clip, m4(width/2), m4(height/2)), 11).resize.Bilinear(width,height)
        elif SrchClipPP > 1:
            # TODO - use gauss_blur from rgtools instead of TCanny?
            # That's what QTGMC in havsfunc uses
            spatialBlur = clip.tcanny.TCanny(sigma=2, mode=-1, planes=CMplanes)
            spatialBlur = spatialBlur.std.Merge(clip, [0.1] if self.ChromaMotion or isGRAY else [0.1, 0])
        else:
            spatialBlur = clip

        if SrchClipPP < 3:
            srchClip = spatialBlur
        else:
            expr = 'x {a} + y < x {b} + x {a} - y > x {b} - x y + 2 / ? ?'.format(a=7*bitDepthMultiplier, b=2*bitDepthMultiplier)
            srchClip = core.std.Expr([spatialBlur, clip], [expr] if ChromaMotion or isGRAY else [expr, ''])

        # TODO: Maybe leave this at default... 
        # Although the defaults may be assuming that rec doesn't use
        # progressively smaller blocksizes automatically, but it does, at least
        # to a degree
        refine = 1 if self.rec else 0
        mv1 = MVTools(clip, tr=degrainTR, refine=refine, super_args=super_args, prefilter=srchClip)

        if degrainTR > 0:
            s2 = limitSigma * 0.625
            s3 = limitSigma * 0.375
            s4 = limitSigma * 0.250
            ovNum = [4, 4, 4, 3, 2, 2][grainLevel]
            ov = 2 * round(limitBlksz / ovNum * 0.5)

            # TODO: Allow custom limit filters.

            if hasattr(core, 'neo_fft3d'):
              spat = core.neo_fft3d.FFT3D(clip, planes=fPlane, sigma=limitSigma, sigma2=s2, sigma3=s3, sigma4=s4, bt=3, bw=limitBlksz, bh=limitBlksz, ow=ov, oh=ov, ncpu=fftThreads)
            else:
              spat = core.fft3dfilter.FFT3DFilter(clip, planes=fPlane, sigma=limitSigma, sigma2=s2, sigma3=s3, sigma4=s4, bt=3, bw=limitBlksz, bh=limitBlksz, ow=ov, oh=ov, ncpu=fftThreads)
            spatD  = core.std.MakeDiff(clip, spat)

        # First MV-denoising stage. Usually here's some temporal-medianfiltering going on.
        NR1 = mv1.degrain(thSAD=thSAD1, thSCD=(thSCD1, thSCD2))

        # Limit NR1 to not do more than what "spat" would do.
        if degrainTR > 0:
            NR1D = core.std.MakeDiff(clip, NR1)
            expr = 'x abs y abs < x y ?' if isFLOAT else f'x {neutral} - abs y {neutral} - abs < x y ?'
            DD   = core.std.Expr([spatD, NR1D], [expr])
            NR1x = core.std.MakeDiff(clip, DD, [0])

            # TODO Add plane configuration
            # TODO Add thSADC support, like AVS version
            mv2 = MVTools(NR1x, tr=degrainTR, refine=refine, super_args=super_args, prefilter)
            NR2 = mv2.degrain(thSAD=thSAD2, thSCD=(thSCD1, thSCD2))
        else
            NR2 = clip

        # TODO Add postTr/postFFT support.
        # TODO update MVTools params
        # TODO Use partial here to setup reusable functions for post FFT that can
        # be used on compensated and non-compensated output easily.
        if postTR > 0:
            mvNoiseWindow = MVTools(NR2)



foo = TemporalDegrain2()
