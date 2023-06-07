"""Representation utilities."""
import pathlib
import pprint

import muspy
import numpy as np

import utils

# Configuration
RESOLUTION = 12
# MAX_BEAT = 1024
MAX_DURATION = 384
MAX_TEMPO = 240
MAX_BAR = 64

# Duration
KNOWN_DURATIONS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    15,
    16,
    18,
    20,
    21,
    24,
    30,
    36,
    40,
    42,
    48,
    60,
    72,
    84,
    96,
    120,
    144,
    168,
    192,
    384,
]
DURATION_MAP = {
    i: KNOWN_DURATIONS[np.argmin(np.abs(np.array(KNOWN_DURATIONS) - i))]
    for i in range(1, MAX_DURATION + 1)
}

KNOWN_VELOCITIES = [16, 32, 48, 64, 80, 96, 112]

VELOCITY_MAP = {
    i: int(max(1, min(7, round(i / 16))) * 16)
    for i in range(1,128)
}

## Instrument
PROGRAM_INSTRUMENT_MAP = [
    # Pianos
    "grand-piano",
    "bright-piano",
    "electric-grand-piano",
    "honky-tony-piano",
    "electric-piano-1",
    "electric-piano-2",
    "harpsichord",
    "clavinet",
    # Chromatic Percussion
    "celesta",
    "glockenspiel",
    "music-box",
    "vibraphone",
    "marimba",
    "xylophone",
    "tubular-bells",
    "dulcimer",
    # Organs
    "dwawbar-organ",
    "percussive-organ",
    "rock-organ",
    "church-organ",
    "reed-organ",
    "accordion",
    "harmonica",
    "bandoneon",
    # Guitars
    "nylon-string-guitar",
    "steel-string-guitar",
    "jazz-electric-guitar",
    "clean-electric-guitar",
    "muted-electric-guitar",
    "overdriven-electric-guitar",
    "distort-electric-guitar",
    "guitar-harmonic",
    # Basses
    "bass",
    "finger-electric-bass",
    "pick-electric-bass",
    "fretless-electric-bass",
    "slap-bass-1",
    "slap-bass-2",
    "synth-bass-1",
    "synth-bass-2",
    # Strings
    "violin",
    "viola",
    "cello",
    "contrabass",
    "tremelo-strings",
    "pizzicato-strings",
    "harp",
    "timpani",
    # Ensemble
    "strings",
    "strings",
    "synth-strings-1",
    "synth-strings-2",
    "voices-aah",
    "voices-ooh",
    "synth-voice",
    "orchestra-hit",
    # Brass
    "trumpet",
    "trombone",
    "tuba",
    "muted-trumpet",
    "horn",
    "brasses",
    "synth-brasses-1",
    "synth-brasses-2",
    # Reed
    "soprano-saxophone",
    "alto-saxophone",
    "tenor-saxophone",
    "baritone-saxophone",
    "oboe",
    "english-horn",
    "bassoon",
    "clarinet",
    # Pipe
    "piccolo",
    "flute",
    "recorder",
    "pan-flute",
    "blown-bottle",
    "Shakuhachi",
    "Whistle",
    "ocarina",
    # Synth Lead
    "lead-square",
    "lead-sawtooth",
    "lead-calliope",
    "lead-chiff",
    "lead-charang",
    "lead-voice",
    "lead-fifths",
    "lead-bass+lead",
    # Synth Pad
    "pad-new-age",
    "pad-warm",
    "pad-polysynth",
    "pad-choir",
    "pad-bowed",
    "pad-metallic",
    "pad-halo",
    "pad-sweep",
    # Synth Effects
    "fx-rain",
    "fx-soundtrack",
    "fx-crystal",
    "fx-atmosphere",
    "fx-brightness",
    "fx-goblins",
    "fx-echoes",
    "fx-scifi",
    # Ethnic
    "sitar",
    "banjo",
    "shamisen",
    "koto",
    "kalimba",
    "bag-pipe",
    "violin",
    "shehnai",
    # Percussive
    "tinkle-bell",
    "agogo",
    "steel-drum",
    "woodblock",
    "taiko",
    "melodic-tom",
    "synth-drums",
    "reverse-cymbal",
    "guitar-fret-noise",
    # Sound effects
    "breath-noise",
    "seashore",
    "bird-tweet",
    "telephone-rang",
    "helicopter",
    "applause",
    "gunshot",
    "drumset",
]
INSTRUMENT_PROGRAM_MAP = {
    instrument: program
    for program, instrument in enumerate(PROGRAM_INSTRUMENT_MAP)
}
PROGRAM_INSTRUMENT_MAP = {
    program: instrument
    for program, instrument in enumerate(PROGRAM_INSTRUMENT_MAP)
}
KNOWN_PROGRAMS = list(
    k for k, v in INSTRUMENT_PROGRAM_MAP.items() if v is not None
)
KNOWN_INSTRUMENTS = list(dict.fromkeys(INSTRUMENT_PROGRAM_MAP.keys()))

def quatize_tempo(qpm):
    return max(7.5, min(MAX_TEMPO, round(qpm / 7.5) * 7.5))

KNOWN_TEMPO = [7.5*(i+1) for i in range(32)]

TEMPO_MAP = {
    i: max(7.5, min(MAX_TEMPO, round(i / 7.5) * 7.5))
    for i in range(1, MAX_TEMPO + 1)
}

MAX_NUMERATOR = 12
KNOWN_TIME_SIGNATURE = [
    f'{p}/{q}'
    for q in [2, 4, 8, 16]
    for p in range(1, MAX_NUMERATOR + 1)
]

KNOWN_EVENTS = [
    "start-of-song",
    "end-of-song",
]
KNOWN_EVENTS.extend(f"bar_{i}" for i in range(1,MAX_BAR+1))
KNOWN_EVENTS.extend(f"time-signature_{i}" for i in KNOWN_TIME_SIGNATURE)
KNOWN_EVENTS.extend(f"position_{i}" for i in range(2*MAX_NUMERATOR*RESOLUTION))
KNOWN_EVENTS.extend(
    f"instrument_{instrument}" for instrument in KNOWN_INSTRUMENTS
)
KNOWN_EVENTS.extend(f'tempo_{i:.1f}' for i in KNOWN_TEMPO)
KNOWN_EVENTS.extend(f"pitch_{i}" for i in range(128))
KNOWN_EVENTS.extend(f"velocity_{i}" for i in KNOWN_VELOCITIES)
KNOWN_EVENTS.extend(f"duration_{i}" for i in KNOWN_DURATIONS)
EVENT_CODE_MAPS = {event: i for i, event in enumerate(KNOWN_EVENTS)}
CODE_EVENT_MAPS = utils.inverse_dict(EVENT_CODE_MAPS)


class Indexer:
    def __init__(self, data=None, is_training=False):
        self._dict = dict() if data is None else data
        self._is_training = is_training

    def __getitem__(self, key):
        if self._is_training and key not in self._dict:
            self._dict[key] = len(self._dict)
            return len(self._dict) - 1
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __contain__(self, item):
        return item in self._dict

    def get_dict(self):
        """Return the internal dictionary."""
        return self._dict

    def train(self):
        """Set training mode."""
        self._is_training = True

    def eval(self):
        """Set evaluation mode."""
        self._is_learning = False


def get_encoding():
    """Return the encoding configurations."""
    return {
        "resolution": RESOLUTION,
        "max_bar": MAX_BAR,
        "max_duration": MAX_DURATION,
        "max_tempo": MAX_TEMPO,
        "time_signature_map": KNOWN_TIME_SIGNATURE,
        "program_instrument_map": PROGRAM_INSTRUMENT_MAP,
        "instrument_program_map": INSTRUMENT_PROGRAM_MAP,
        "tempo_map": TEMPO_MAP,
        "duration_map": DURATION_MAP,
        "velocity_map": VELOCITY_MAP,
        "event_code_map": EVENT_CODE_MAPS,
        "code_event_map": CODE_EVENT_MAPS,
    }


def load_encoding(filename):
    """Load encoding configurations from a JSON file."""
    encoding = utils.load_json(filename)
    for key in ("program_instrument_map", "code_event_map", "duration_map", "velocity_map", "tempo_map"):
        encoding[key] = {
            int(k) if k != "null" else None: v
            for k, v in encoding[key].items()
        }
    return encoding


def extract_notes(music, resolution):
    """Return a MusPy music object as a note sequence.

    Each row of the output is a note specified as follows.

        (onset, pitch, duration, program, velocity)

    """
    # Check resolution
    assert music.resolution == resolution
    assert all([(ts.numerator == 3 or ts.numerator == 4) and ts.denominator == 4 for ts in music.time_signatures])

    # Extract notes
    notes = []
    for track in music:
        for note in track:
            notes.append(
                (note.time, note.pitch, note.duration, track.program, note.velocity)
            )

    # Deduplicate and sort the notes
    notes = sorted(set(notes))

    return np.array(notes)


# def encode_notes(notes, encoding, indexer):
#     """Encode the notes into a sequence of code tuples.

#     Each row of the output is encoded as follows.

#         (event_type, beat, position, pitch, duration, instrument)

#     """
#     # Get variables
#     max_beat = encoding["max_beat"]
#     max_duration = encoding["max_duration"]

#     # Get maps
#     duration_map = encoding["duration_map"]
#     program_instrument_map = encoding["program_instrument_map"]
#     velocity_map = encoding["velocity_map"]

#     # Start the codes with an SOS event
#     codes = [indexer["start-of-song"]]

#     # Encode the notes
#     last_beat = 0
#     for beat, position, pitch, duration, program, velocity in notes:
#         # Skip if max_beat has reached
#         if beat > max_beat:
#             continue
#         # Skip unknown instruments
#         instrument = program_instrument_map[program]
#         if instrument is None:
#             continue
#         if beat > last_beat:
#             codes.append(indexer[f"beat_{beat}"])
#             last_beat = beat
#         codes.append(indexer[f"position_{position}"])
#         codes.append(indexer[f"instrument_{instrument}"])
#         codes.append(indexer[f"pitch_{pitch}"])
#         codes.append(
#             indexer[f"duration_{duration_map[min(duration, max_duration)]}"]
#         )
#         codes.append(indexer[f"velocity_{velocity_map[velocity]}"])

#     # End the codes with an EOS event
#     codes.append(indexer["end-of-song"])

#     return np.array(codes)

BAR_ORDER = 0
TIMESIG_ORDER = 1
TEMPO_ORDER = 2
NOTE_ORDER = 3

def encode(music, encoding, indexer):
    """Encode a MusPy music object into a sequence of codes.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Extract notes
    # notes = extract_notes(music, encoding["resolution"])

    # # Encode the notes
    # codes = encode_notes(notes, encoding, indexer)

    resolution = encoding['resolution']
    assert music.resolution == resolution

    time_signature_map = set(encoding['time_signature_map'])
    max_bar = encoding['max_bar']
    assert all([
        f'{ts.numerator}/{ts.denominator}' in time_signature_map
        for ts in music.time_signatures
    ])

    if len(music.time_signatures) == 0:
        music.time_signatures.append(muspy.TimeSignature(0, 4, 4))
    else:
        music.time_signatures.sort()
        assert music.time_signatures[0].time == 0

    if len(music.tempos) == 0:
        music.tempos.sort()
        music.tempos.append(muspy.Tempo(0, 120))
    else:
        assert music.tempos[0].time == 0

    end_time = music.get_end_time()

    # Make bar & timesig events
    bar_num = 1
    bar_event_list = []
    timesig_event_list = []
    timesig_cursor = 0
    cur_bar_start_time = 0
    cur_bar_length = 4 * music.time_signatures[0].numerator * resolution // music.time_signatures[0].denominator
    next_bar_start_time = cur_bar_length
    if len(music.time_signatures) > 1:
        next_timesig_start_time = music.time_signatures[1].time
    else:
        next_timesig_start_time = end_time + cur_bar_length # never reach
    while cur_bar_start_time < end_time and bar_num < max_bar:
        bar_event_list.append((cur_bar_start_time, BAR_ORDER, bar_num))
        timesig_event_list.append(
            (cur_bar_start_time,
             TIMESIG_ORDER,
             music.time_signatures[timesig_cursor].numerator,
             music.time_signatures[timesig_cursor].denominator)
        )
        cur_bar_start_time += cur_bar_length
        bar_num += 1
        if cur_bar_start_time == next_timesig_start_time:
            timesig_cursor += 1
            cur_bar_length = 4 * music.time_signatures[timesig_cursor].numerator * resolution // music.time_signatures[timesig_cursor].denominator
            if len(music.time_signatures) > timesig_cursor+1:
                next_timesig_start_time = music.time_signatures[timesig_cursor+1].time
                assert (next_timesig_start_time - cur_bar_start_time) % cur_bar_length == 0
            else:
                next_timesig_start_time = end_time + cur_bar_length # never reach
        next_bar_start_time += cur_bar_length

    if next_bar_start_time < end_time:
        end_time = next_bar_start_time

    # Make tempo events
    tempo_event_list = []
    tempo_cursor = 0
    bar_cursor = 0
    if len(music.tempos) > 1:
        next_tempo_start_time = music.tempos[1].time
    else:
        next_tempo_start_time = end_time + 1
    while bar_cursor < len(bar_event_list) or next_tempo_start_time < end_time:
        if bar_cursor < len(bar_event_list) and bar_event_list[bar_cursor][0] < next_tempo_start_time:
            tempo_event_list.append((bar_event_list[bar_cursor][0], TEMPO_ORDER, music.tempos[tempo_cursor].qpm))
            bar_cursor += 1
        else:
            tempo_cursor += 1
            tempo_event_list.append((music.tempos[tempo_cursor].time, TEMPO_ORDER, music.tempos[tempo_cursor].qpm))
            if len(music.tempos) > tempo_cursor+1:
                next_tempo_start_time = music.tempos[tempo_cursor+1].time
            else:
                next_tempo_start_time = end_time + 1

    # Make note events
    note_event_list = []
    for track in music:
        for note in track:
            program = 128 if track.is_drum else track.program
            if note.time < end_time:
                note_event_list.append((note.time, NOTE_ORDER, program, note.pitch, note.velocity, note.duration))

    all_event_list = bar_event_list + timesig_event_list + tempo_event_list + note_event_list
    all_event_list.sort()

    max_tempo = encoding['max_tempo']
    tempo_map = encoding['tempo_map']
    program_instrument_map = encoding['program_instrument_map']
    velocity_map = encoding['velocity_map']
    max_duration = encoding['max_duration']
    duration_map = encoding['duration_map']

    codes = ['start-of-song']
    cur_bar_start_time = 0
    for event in all_event_list:
        order = event[1]
        if order == BAR_ORDER:
            codes.append(f'bar_{event[2]}')
            cur_bar_start_time = event[0]
        elif order == TIMESIG_ORDER:
            codes.append(f'time-signature_{event[2]}/{event[3]}')
        elif order == TEMPO_ORDER:
            codes.append(f'position_{event[0]-cur_bar_start_time}')
            qpm = tempo_map[min(max_tempo, round(event[2]))]
            codes.append(f'tempo_{qpm:.1f}')
        elif order == NOTE_ORDER:
            # assert event[0]-cur_bar_start_time < end_time
            codes.append(f'position_{event[0]-cur_bar_start_time}')
            instrument = program_instrument_map[event[2]]
            velocity = velocity_map[event[4]]
            duration = duration_map[min(max_duration, event[5])]
            codes.append(f'instrument_{instrument}')
            codes.append(f'pitch_{event[3]}')
            codes.append(f'velocity_{velocity}')
            codes.append(f'duration_{duration}')
    codes.append('end-of-song')
    codes = np.array(list(map(indexer.__getitem__, codes)), dtype=np.int32)
    return codes


def decode_notes(data, encoding, vocabulary):
    """Decode codes into a note sequence."""
    # Get variables and maps
    instrument_program_map = encoding["instrument_program_map"]

    # Initialize variables
    cur_bar_start_time = None
    cur_bar_length = None
    # bar_num = None
    position = None
    program = None
    pitch = None
    duration = None
    velocity = None

    tempos = []
    time_signatures = []

    # Decode the codes into a sequence of notes
    notes = []
    for code in data:
        event = vocabulary[code]
        if event == "start-of-song":
            continue
        elif event == "end-of-song":
            break
        elif event.startswith("bar"):
            # bar_num = int(event.split("_")[1])
            if cur_bar_start_time is None:
                cur_bar_start_time = 0
            else:
                cur_bar_start_time += cur_bar_length
            # Reset variables
            position = None
            program = None
            pitch = None
            velocity = None
            duration = None
        elif event.startswith("time-signature"):
            timesig = event.split("_")[1].split("/")
            n, d = int(timesig[0]), int(timesig[1])
            if len(time_signatures) == 0 or time_signatures[-1][1] != n:
                cur_bar_length = 4 * n * encoding['resolution'] // d
                time_signatures.append((cur_bar_start_time, n))
        elif event.startswith("position"):
            position = int(event.split("_")[1])
            # Reset variables
            program = None
            pitch = None
            velocity = None
            duration = None
        elif event.startswith("tempo"):
            tempo = float(event.split("_")[1])
            if len(tempos) == 0 or tempos[-1][1] != tempos:
                tempos.append((cur_bar_start_time+position, tempo))
        elif event.startswith("instrument"):
            instrument = event.split("_")[1]
            program = instrument_program_map[instrument]
        elif event.startswith("pitch"):
            pitch = int(event.split("_")[1])
        elif event.startswith("velocity"):
            velocity = int(event.split("_")[1])
        elif event.startswith("duration"):
            duration = int(event.split("_")[1])
            if (
                position is None
                or program is None
                or pitch is None
                or duration is None
                or velocity is None
            ):
                continue
            notes.append((cur_bar_start_time+position, program, pitch, velocity, duration))
        else:
            raise ValueError(f"Unknown event type for: {event}")

    return notes, tempos, time_signatures


def reconstruct(notes, tempos, time_signatures, resolution):
    """Reconstruct a note sequence to a MusPy Music object."""
    # Construct the MusPy Music object
    music = muspy.Music(
        resolution=resolution,
        tempos=[muspy.Tempo(time, qpm) for time, qpm in tempos],
        time_signatures=[muspy.TimeSignature(time, n, 4) for time, n in time_signatures]
    )

    # Append the tracks
    programs = sorted(set(note[1] for note in notes))
    for program in programs:
        if program == 128:
            music.tracks.append(muspy.Track(is_drum=True))
        else:
            music.tracks.append(muspy.Track(program))

    # Append the notes
    for time, program, pitch, velocity, duration in notes:
        track_idx = programs.index(program)
        music[track_idx].notes.append(muspy.Note(time, pitch, duration, velocity))

    return music


def decode(codes, encoding, vocabulary):
    """Decode codes into a MusPy Music object.

    Each row of the input is encoded as follows.

        (event_type, beat, position, pitch, duration, instrument)

    """
    # Get resolution
    resolution = encoding["resolution"]

    # Decode codes into a note sequence
    notes, tempos, time_signatures = decode_notes(codes, encoding, vocabulary)

    # Reconstruct the music object
    music = reconstruct(notes, tempos, time_signatures, resolution)

    return music


def dump(data, vocabulary):
    """Decode the codes and dump as a string."""
    # Iterate over the rows
    lines = []
    for code in data:
        event = vocabulary[code]
        if (
            event == "start-of-song"
            or event.startswith("bar")
            or event.startswith("position")
        ):
            lines.append(event)
        elif event.startswith("time-signature"):
            lines[-1] = f"{lines[-1]} {event}"
        elif event == "end-of-song":
            lines.append(event)
            break
        elif (
            event.startswith("tempo")
            or event.startswith("instrument")
            or event.startswith("pitch")
            or event.startswith("velocity")
            or event.startswith("duration")
        ):
            lines[-1] = f"{lines[-1]} {event}"
        else:
            raise ValueError(f"Unknown event type for: {event}")

    return "\n".join(lines)


def save_txt(filename, data, vocabulary):
    """Dump the codes into a TXT file."""
    with open(filename, "w") as f:
        f.write(dump(data, vocabulary))


def save_csv_notes(filename, data):
    """Save the representation as a CSV file."""
    assert data.shape[1] == 5
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="beat,position,pitch,duration,program,velocity",
        comments="",
    )


def save_csv_codes(filename, data):
    """Save the representation as a CSV file."""
    assert data.ndim == 1
    np.savetxt(
        filename,
        data,
        fmt="%d",
        delimiter=",",
        header="code",
        comments="",
    )


def main():
    """Main function."""
    # Get the encoding
    encoding = get_encoding()

    # Save the encoding
    filename = pathlib.Path(__file__).parent / "encoding_remi.json"
    utils.save_json(filename, encoding)

    # Load encoding
    encoding = load_encoding(filename)

    # Print the maps
    print(f"{' Maps ':=^40}")
    for key, value in encoding.items():
        if key in ("program_instrument_map", "instrument_program_map"):
            print("-" * 40)
            print(f"{key}:")
            pprint.pprint(value, indent=2)

    # Print the variables
    print(f"{' Variables ':=^40}")
    print(f"resolution: {encoding['resolution']}")
    print(f"max_bar: {encoding['max_bar']}")
    print(f"max_duration: {encoding['max_duration']}")
    print(f"max_tempo: {encoding['max_tempo']}")

    # Load the example
    music = muspy.load(pathlib.Path(__file__).parent / "example_remi.json")

    # Get the indexer
    indexer = Indexer(is_training=True)

    # Encode the music
    encoded = encode(music, encoding, indexer)
    print(f"Codes:\n{encoded}")

    # Get the learned vocabulary
    vocabulary = utils.inverse_dict(indexer.get_dict())

    print("-" * 40)
    print(f"Decoded:\n{dump(encoded, vocabulary)}")

    music = decode(encoded, encoding, vocabulary)
    print(f"Decoded musics:\n{music}")


if __name__ == "__main__":
    main()
