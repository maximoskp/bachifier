import music21 as m21
import numpy as np
import os
import pandas as pd

class SymbolicInfo:
    
    metadata = None
    
    def __init__(self, filepath, metadatafile=None, logging=False, truncation=True, make12t=True):
        if logging:
            print('processing ', filepath)
        if metadatafile is not None and SymbolicInfo.metadata is None:
            SymbolicInfo.metadata = pd.read_csv( metadatafile )
        if filepath.split('.')[-1] in ['xml', 'mid', 'midi', 'mxl', 'musicxml']:
            self.name = filepath.split('.')[-2].split(os.sep)[-1]
            self.truncation = truncation
            if SymbolicInfo.metadata is not None:
                self.title = self.metadata[ self.metadata['ID'] == self.name ]['Title']
            self.stream = m21.converter.parse( filepath )
            self.flat = self.stream.flat.notes
            self.pcs = []
            self.pcp = np.zeros(12)
            self.make_pcp()
            self.estimate_tonality()
            self.make_NOD_string()
            if make12t:
                print('making 12t')
                self.make_12t_NOD_string()
        else:
            print('bad format')
    # end __init__

    def make_pcs(self):
        for p in self.flat.pitches:
            self.pcs.append( p.midi%12 )
    # end make_pcp
    
    def make_pcp(self):
        self.make_pcs()
        for p in self.pcs:
            self.pcp[p] += 1
        if np.sum(self.pcp) != 0:
            self.pcp = self.pcp/np.sum(self.pcp)
    # end make_pcp
    
    def estimate_tonality(self):
        p = m21.analysis.discrete.KrumhanslSchmuckler()
        self.estimated_tonality = p.getSolution(self.stream)
    # end estimate_tonality
    
    def make_NOD_string(self):
        s = ''
        prev_offset = 0
        for n in self.flat.notes:
            if isinstance(n, m21.note.Note):
                if n.offset - prev_offset >= 0:
                    offset_string = str(float(n.offset) - prev_offset)
                    duration_string = str(float(n.duration.quarterLength))
                    if self.truncation:
                        offset_string = '{:.2f}'.format( float(n.offset) - prev_offset )
                        duration_string = '{:.2f}'.format( float(n.duration.quarterLength) )
                    s += '_(' + str(n.pitch.midi) + ',' + offset_string + ',' + duration_string + ')'
                    prev_offset = float(n.offset)
            elif isinstance(n, m21.chord.Chord):
                if n.offset - prev_offset >= 0:
                    for note in n:
                        offset_string = str(float(n.offset) - prev_offset)
                        duration_string = str(float(n.duration.quarterLength))
                        if self.truncation:
                            offset_string = '{:.2f}'.format( float(n.offset) - prev_offset )
                            duration_string = '{:.2f}'.format( float(n.duration.quarterLength) )
                        s += '_(' + str(note.pitch.midi) + ',' + offset_string + ',' + duration_string + ')'
                    prev_offset = float(n.offset)
        self.nod_string = s
    # end make_NOD_string

    def make_12t_NOD_string(self):
        s = ''
        for i in range(-6, 6, 1):
            ival = m21.interval.Interval( i )
            st = self.stream.transpose(ival)
            f = st.flat
            prev_offset = 0
            for n in f.notes:
                if isinstance(n, m21.note.Note):
                    if n.offset - prev_offset >= 0:
                        offset_string = str(float(n.offset) - prev_offset)
                        duration_string = str(float(n.duration.quarterLength))
                        if self.truncation:
                            offset_string = '{:.2f}'.format( float(n.offset) - prev_offset )
                            duration_string = '{:.2f}'.format( float(n.duration.quarterLength) )
                        s += '_(' + str(n.pitch.midi) + ',' + offset_string + ',' + duration_string + ')'
                        prev_offset = float(n.offset)
                elif isinstance(n, m21.chord.Chord):
                    if n.offset - prev_offset >= 0:
                        for note in n:
                            offset_string = str(float(n.offset) - prev_offset)
                            duration_string = str(float(n.duration.quarterLength))
                            if self.truncation:
                                offset_string = '{:.2f}'.format( float(n.offset) - prev_offset )
                                duration_string = '{:.2f}'.format( float(n.duration.quarterLength) )
                            s += '_(' + str(note.pitch.midi) + ',' + offset_string + ',' + duration_string + ')'
                        prev_offset = float(n.offset)
        self.nod_12t_string = s
    # end make_NOD_string
# end class SymbolicInfo


def stream_from_NOD_string(s):
    usplit = s.split('_')
    s = m21.stream.Score()
    tm = m21.tempo.MetronomeMark(number=80)
    s.insert(0, tm)
    total_offset = 0
    for u in usplit:
        csplit = u[1:-1].split(',')
        if len( csplit ) == 3:
            n = m21.note.Note(int(csplit[0]))
            d = m21.duration.Duration(float(csplit[2]))
            n.duration = d
            # n.show('t')
            total_offset += float( csplit[1] )
            s.insert( total_offset , n )
    return s
# end stream_from_NOD_string

def generate_midi(sc, fileName="test_midi.mid"):
    # we might want the take the name from uData information, e.g. the uData.input_id, which might preserve a unique key for identifying which file should be sent are response to which user
    mf = m21.midi.translate.streamToMidiFile(sc)
    mf.open(fileName, 'wb')
    mf.write()
    mf.close()
# end generate_midi

def midi_from_NOD_string(s, fileName="test_midi.mid"):
    m21s = stream_from_NOD_string(s)
    generate_midi(m21s, fileName=fileName)
# end midi_from_NOD_string