# Fx Classifier Doc
### Version 0.1 -- 20/05/2022

## Input:

- `audio`: `array` of shape (num_channels, num_samples) representing the audio file. The file can be stereo but 
it has no impact on the treatment;
- `rate`: `int`, sampling rate of the audio file (22 050Hz is often enough).

Audio of a note played with the target Fx. It should be a single note of any duration.

## Output:

- `cls`: `int` representing the  Fx detected according to the following chart:
  
| `cls` |    Fx Type     |                       Plugin to instanciate                        |
|:-----:|:--------------:|:------------------------------------------------------------------:|
|  `0`  |      Dry       |                                None                                |
|  `1`  | Feedback Delay |                          `dsp::DelayLine`                          |
|  `2`  | Slapback Delay |                          `dsp::DelayLine`                          |
|  `3`  |     Reverb     |                           `dsp::Reverb`                            |
|  `4`  |     Chorus     |                           `dsp::Chorus`                            |
|  `5`  |    Flanger     |                           `dsp::Chorus`                            |
|  `6`  |     Phaser     |                           `dsp::Chorus`                            |
|  `7`  |    Tremolo     |           _probably_ <br/>`[dsp::Oscillator, dsp::Gain]`           |
|  `8`  |    Vibrato     |                           `dsp::Chorus`                            |
|  `9`  |   Distortion   | `[dsp::Gain, dsp::WaveShaper, dsp::IIR::Filter, dsp::IIR::Filter]` |
| `10`  |   Overdrive    |                               _idem_                               |


### References:

- https://github.com/spotify/pedalboard
- https://docs.juce.com/master/group__juce__dsp.html
- https://github.com/adhooge/AutoFX
- Stein et al., _Automatic Detection of Audio Effects in Guitar and Bass Recordings_, 2010