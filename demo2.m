% This script shows a very simple example of music transcription with NMF
% We load a clip of a piano playing some notes (with polyphony), create a
% magnitude spectrogram, and then do NMF on the result.  The columns of W 
% can be interpreted as the spectra of the notes (actually of the unique
% 'objects' that occur - see below) and the rows of H give the activations
% of those notes over time, thus functioning much like a pianoroll or score.

% Spectrogram parameters
nfft = 512;
win = 0.064; % seconds
hop = 0.032; % seconds

% First load the data (this is a simple synthesized piano sequence)
[d,sr] = wavread('data/example.wav');

% Now make a magnitude spectrogram
V = abs(spectrogram(d, ceil(win*sr), ceil(hop*sr), nfft));

% We know the clip has 6 unique pitches in it, so we'll set the rank to that
[W,H,err] = nmf_alg(V, 6, 'alg', @nmf_kl, 'niter', 100, 'verb', 2);

% Note that the order of the components in W and H is arbitrary so we'd 
% have to sort them manually to make H easy to compare to the pianoroll,   
% which we do not do here.  Notice that NMF has identified and separated 
% only those notes that do not always co-occur.  In the example, two of the
% notes (the D4 and A4 at the 4th beat) only occur once each and only at 
% the same point.  Thus as far as NMF is concerned, they are the same thing 
% and so get assigned to a single component. This means that we have one 
% extra component (since there are 5 unique objects) which generally 
% corresponds to noise and so is not much of an issue.

% Do some plotting
figure; plot(err); xlabel('Iterations'); ylabel('Error');
figure; imagesc(W); axis xy; xlabel('Component'); ylabel('Frequency');
figure; imagesc(H); axis xy; xlabel('Time'); ylabel('Component');

% If we have the MIDI Toolbox installed, we can load the MIDI file and plot
% the pianoroll for comparison.  Otherwise, there is a jpg of the pianoroll
% included as well.
if exist('readmidi')
    nmat = readmidi('data/example.mid');
    figure; pianoroll(nmat);
end

