import struct
import os


class NotMP4FormatException(Exception):
    pass


class ATOM(object):
    def __init__(self, size, name, pos):
        self.size = size
        self.name = name
        self.pos = pos
        self.children = []
        self.properties = None

    def find_child_atom_internal(self, atoms, part_arr):
        name = part_arr[0]
        for atom in atoms:
            if atom.name == name:
                if len(part_arr) == 1:
                    return atom

                return self.find_child_atom_internal(atom.children, part_arr[1:])

        return None

    def find_child_atom(self, name):
        part_arr = name.split("/")
        return self.find_child_atom_internal(self.children, part_arr)

    def __str__(self):
        return "%s(%s)" % (self.name, self.size)

    def __repr__(self):
        return self.__str__()


class CHUNK(object):
    def __init__(self, pos):
        self.pos = pos
        self.samples_count = 0
        self.samples_desc_idx = 0
        self.size = 0

    def __str__(self):
        return "CHUNK(pos: %s, size: %s, samples: %s)" % (self.pos, self.size, self.samples_count)

    def __repr__(self):
        return self.__str__()


class Frame(object):
    def __init__(self):
        self.pos = -1
        self.type = 0
        self.size = 0

    def __str__(self):
        return "Frame(pos: %s, size: %s, type: %s)" % (self.pos, self.size, self.type)

    def __repr__(self):
        return self.__str__()


class TRACK(object):
    def __init__(self, track_idx, track_atom):
        self.track_idx = track_idx
        self.track_atom = track_atom
        self.stbl_atom = None
        self.ppss = None
        self.spss = None
        self.chunks = []
        self.build()

    def __str__(self):
        return "TRACK(%s)" % (self.track_idx)

    def __repr__(self):
        return self.__str__()

    def build(self):
        self.stbl_atom = self.track_atom.find_child_atom("mdia/minf/stbl")

        stsd_atom = self.stbl_atom.find_child_atom("stsd")
        if stsd_atom is None:
            return

        stco_atom = self.stbl_atom.find_child_atom("stco")
        if stco_atom is None:
            stco_atom = self.stbl_atom.find_child_atom("co64")

        chunks = []

        for stco in stco_atom.properties:
            chunks.append(CHUNK(stco))

        self.chunks = chunks
        stsc_atom = self.stbl_atom.find_child_atom("stsc")

        chunks_samples = []
        end_chunk = len(stco_atom.properties) - 1
        for i, entry in enumerate(stsc_atom.properties):
            start_chunk = entry['first_chunk'] - 1
            sample_count = entry['samples_per_chunk']
            desc_idx = entry['sample_description_index']

            if i != 0:  # update end trunk
                chunks_samples[-1][1] = start_chunk - 1

            chunks_samples.append([start_chunk, end_chunk, sample_count, desc_idx])

        idx = 0
        sample_info = chunks_samples[idx]
        chunks_samples_len = len(chunks_samples)
        for i in range(0, len(stco_atom.properties)):
            chunk = chunks[i]
            if sample_info[1] < i:
                idx += 1
                if idx >= chunks_samples_len:
                    print('error', idx, len(chunks_samples), chunks_samples)
                else:
                    sample_info = chunks_samples[idx]

            chunk.samples_count = sample_info[2]
            chunk.samples_desc_idx = sample_info[3]
        if len(stsd_atom.properties) > 0:
            self.ppss = stsd_atom.properties[0]["avc"]["pps"]
            self.spss = stsd_atom.properties[0]["avc"]["sps"]


class MP4(object):
    def __init__(self, filename):
        self.filename = filename
        self.f = open(filename, "rb")
        self.children = []
        self.track_size = 0
        self.moov_atoms = None
        self.mdat_atoms = None
        self.tracks = []

    def create_empty_atom(self):
        return ATOM(0, "", 0)

    def get_moov_atoms(self):
        moovs = []
        for atom in self.children:
            if atom.name == "moov":
                moovs.append(atom)

        if len(moovs) == 0:
            print('no moov')
            raise NotMP4FormatException()
        return moovs

    def get_mdat_atoms(self):
        mdats = []
        for atom in self.children:
            if atom.name == "mdat":
                mdats.append(atom)

        if len(mdats) == 0:
            print('no mdat')
            raise NotMP4FormatException()
        return mdats

    def get_track_size(self):
        return self.track_size

    def get_track_count_internal(self):
        count = 0
        for moov in self.moov_atoms:
            for atom in moov.children:
                if atom.name == "trak":
                    count += 1

        return count

    def parse(self, start_pos=0):
        # mp4 container follow BIG ENDIAN
        next_pos = start_pos

        try:
            while True:
                atom = self.parse_internal(next_pos)
                if atom is None:
                    break
                self.children.append(atom)
                next_pos += atom.size

        except:
            raise NotMP4FormatException()

        self.moov_atoms = self.get_moov_atoms()
        self.mdat_atoms = self.get_mdat_atoms()
        self.track_size = self.get_track_count_internal()

        self.tracks = self.merge_tracks()
        return True

    def merge_tracks(self):
        tracks = []
        count = 0
        for moov in self.moov_atoms:
            for atom in moov.children:
                if atom.name == "trak":
                    tracks.append(TRACK(count, atom))
                    count += 1

        return tracks

    def traverse(self, udf=None):
        self.traverse_internal(self.children, 0, udf)

    def traverse_internal(self, atoms, depth, udf=None):
        buf = ""
        for i in range(depth):
            buf += "    "

        for atom in atoms:
            print("%s%s" % (buf, atom))
            if udf is not None:
                udf(atom)

            self.traverse_internal(atom.children, depth + 1, udf)

    def get_atom(self, pos):
        self.f.seek(pos)
        rbytes = self.f.read(4)
        if rbytes is None or len(rbytes) == 0:
            return None
        size = struct.unpack('>I', rbytes)[0]

        name = self.f.read(4)
        #print(name, pos, size)
        name = name.decode('utf-8')

        if size == 1:
            rbytes = self.f.read(8)
            size = int(struct.unpack('>Q', rbytes)[0])
            print('big atom', size, rbytes)
        # elif size == 0 : until end of the file

        return ATOM(size, name, pos)

    def parse_avcc(self, avc, name, size):
        avcC = {}
        spss = []
        ppss = []
        version = struct.unpack('>b', self.f.read(1))[0]
        avc_profile_idc = struct.unpack('>b', self.f.read(1))[0]
        profile_compatibility = struct.unpack('>b', self.f.read(1))[0]
        avc_level_idc = struct.unpack('>b', self.f.read(1))[0]

        lengh_size_minus_one = (struct.unpack('>b', self.f.read(1))[0]) & 0x03 + 1
        num_of_sps = (struct.unpack('>b', self.f.read(1))[0]) & 0x1F
        for i in range(num_of_sps):
            length_sps = struct.unpack('>h', self.f.read(2))[0]
            sps = self.f.read(length_sps)
            spss.append(sps)

        num_of_pps = struct.unpack('>b', self.f.read(1))[0]
        for i in range(num_of_pps):
            length_pps = struct.unpack('>h', self.f.read(2))[0]
            pps = self.f.read(length_pps)
            ppss.append(pps)

        avcC["length_size_minus_one"] = lengh_size_minus_one
        avcC["sps"] = spss
        avcC["pps"] = ppss
        return avcC

    def parse_avc_internal(self, atom):
        avc = {}
        size = struct.unpack('>i', self.f.read(4))[0]
        name = self.f.read(4)
        if name != b"avc1":
            return None

        avc["name"] = name
        self.f.read(24)
        avc["w"] = struct.unpack('>h', self.f.read(2))[0]
        avc["h"] = struct.unpack('>h', self.f.read(2))[0]
        avc["hres"] = struct.unpack('>i', self.f.read(4))[0]
        avc["vres"] = struct.unpack('>i', self.f.read(4))[0]
        self.f.read(4)

        frame_count = struct.unpack('>h', self.f.read(2))[0]
        if frame_count != 1:
            return None

        self.f.read(32)
        depth = struct.unpack('>h', self.f.read(2))[0]
        if depth != 0x18:
            return None

        pd = struct.unpack('>h', self.f.read(2))[0]
        if pd != -1:
            return None

        while True:
            tsize = struct.unpack('>i', self.f.read(4))[0]
            tname = self.f.read(4)

            if tname == b"avcC":
                avc["avc"] = self.parse_avcc(avc, tname, tsize)
                break
            else:
                self.f.read(tsize - 8)

        return avc

    def parse_avc(self, atom):
        self.f.seek(atom.pos + 12)
        entry_count = struct.unpack('>i', self.f.read(4))[0]
        entries = []

        for i in range(entry_count):
            entry = self.parse_avc_internal(atom)
            if entry is not None:
                entries.append(entry)

        return entries

    def parse_internal(self, pos, total_size=0):
        atom = self.get_atom(pos)
        if atom is None:
            return None
        if total_size > 0 and atom.size > total_size:
            return self.create_empty_atom()

        if atom.size < 8 and atom.size != 0:
            print('atom length error', atom.name, atom.pos, atom.size)
            raise NotMP4FormatException

        if atom.name in ['stss', 'stts','ctts', 'mdat', 'tkhd', 'vmhd', "ftyp", "mvhd", "tkhd", "mdhd", "meta", 'smhd']:
            return atom

        if atom.name == "stsd":
            child = self.parse_avc(atom)
            atom.properties = child
            return atom
        elif atom.name == "stsc":
            child = self.parse_stsc(atom)
            atom.properties = child
            return atom
        elif atom.name == "stsz":
            child = self.parse_stsz(atom)
            atom.properties = child
            return atom
        elif atom.name == "stco":
            child = self.parse_stco(atom)
            atom.properties = child
            return atom
        elif atom.name =='co64':
            child = self.parse_co64(atom)
            atom.properties = child
            return atom

        next_pos = atom.pos + 8

        while (next_pos + 8) < (atom.pos + atom.size):
            child = self.parse_internal(next_pos, atom.size)
            if (child.size >= atom.size) or child.size <= 0:
                break

            atom.children.append(child)
            next_pos += child.size

        return atom

    def parse_stsc(self, atom):
        self.f.seek(atom.pos + 12)  # version  and flags
        entry_count = struct.unpack('>i', self.f.read(4))[0]

        entries = []

        for i in range(entry_count):
            first_chunk = struct.unpack('>i', self.f.read(4))[0]
            samples_per_chunk = struct.unpack('>i', self.f.read(4))[0]
            sample_description_index = struct.unpack('>i', self.f.read(4))[0]
            entries.append({'first_chunk': first_chunk, 'samples_per_chunk': samples_per_chunk,
                            'sample_description_index': sample_description_index})

        return entries

    def parse_stsz(self, atom):
        self.f.seek(atom.pos + 16)  # version , flags,  sample_size
        entry_count = struct.unpack('>i', self.f.read(4))[0]
        entry_sizes = []
        for i in range(entry_count):
            sample_size = struct.unpack('>i', self.f.read(4))[0]

            entry_sizes.append(sample_size)

        return entry_sizes

    def parse_stco(self, atom):
        self.f.seek(atom.pos + 12)
        entry_count = struct.unpack('>i', self.f.read(4))[0]
        entries = []

        for i in range(entry_count):
            chunk_offset = struct.unpack('>i', self.f.read(4))[0]
            entries.append(chunk_offset)

        return entries

    def parse_co64(self, atom):
        self.f.seek(atom.pos + 12)

        entry_count = struct.unpack('>i', self.f.read(4))[0]
        entries = []

        for i in range(entry_count):
            chunk_offset = struct.unpack('>Q', self.f.read(8))[0]
            entries.append(chunk_offset)

        return entries

    def get_samples(self):
        frames = []
        video_track = None
        for track in self.tracks:
            stsd_atom = track.stbl_atom.find_child_atom("stsd")
            if len(stsd_atom.properties) > 0 and stsd_atom.properties[0]["avc"] is not None:
                video_track = track
                break

        if video_track is None:
            return frames

        stsz_atom = video_track.stbl_atom.find_child_atom("stsz")

        current_chunk_idx = 0
        current_offset_in_chunk = 0
        current_samplenb_in_chunk = 0
        chunks_in_trak = video_track.chunks
        for (sample_index, sample_size) in enumerate(stsz_atom.properties):
            frame = Frame()
            frames.append(frame)
            frame.size = sample_size
            if current_chunk_idx >= len(chunks_in_trak):
                print('error', sample_index, current_chunk_idx, chunks_in_trak)
                raise NotMP4FormatException
            else:
                frame.pos = chunks_in_trak[current_chunk_idx].pos + current_offset_in_chunk

            current_offset_in_chunk += frame.size
            current_samplenb_in_chunk += 1
            if current_samplenb_in_chunk >= chunks_in_trak[current_chunk_idx].samples_count:
                current_chunk_idx += 1
                current_offset_in_chunk = 0
                current_samplenb_in_chunk = 0

        return frames

    def copy_iframe_data(self, frame, infile_path, outfile_path):

        current_nal_pos  = frame.pos
        is_i_frame = False

        with open(infile_path, "rb") as in_file:
            while current_nal_pos < frame.pos + frame.size and not is_i_frame :
                in_file.seek(current_nal_pos)

                nal_length_bytes = in_file.read(4)
                if nal_length_bytes is None or len(nal_length_bytes) < 4:
                    # nal not complete
                    return None

                nal_length = struct.unpack('>I', nal_length_bytes)[0]
                if nal_length == 1:
                    nal_length_bytes = self.f.read(8)
                    nal_length = int(struct.unpack('>Q', nal_length_bytes)[0])
                    print('big atom', nal_length, nal_length_bytes)

                nal_unit_type_bytes = in_file.read(1)
                if nal_unit_type_bytes is None:
                    return None  # nal not complete
                nal_unit_type = nal_unit_type_bytes[0] & 0x1f

                first_byte_in_slice = in_file.read(1)
                if first_byte_in_slice == None:
                    return None

                if nal_unit_type == 5:  # 5: idr frame is i frame
                    #print("idr frame")
                    is_i_frame = True
                    break
                slice_type_byte = first_byte_in_slice[0] & 0b1111111
                slice_type = MP4.unary_decode(format(slice_type_byte, '07b'))
                #print("nal_unit_type", nal_unit_type, 'slice type', slice_type)
                if slice_type in [2, 7]:
                    is_i_frame = True
                    break

                current_nal_pos += nal_length + len(nal_length_bytes)

        if is_i_frame:
            self.copy_frame_data(frame, infile_path, outfile_path)

        return None

    @staticmethod
    def unary_decode(coding):
        n = 0

        while n < 7 and coding[n] == '0':
            n += 1

        if n == 0:
            return (1 << n) - 1
        return (1 << n) - 1 + int(coding[n + 1:n+1+n], 2)


    def write_i_frame(self, filename, frames, output_file_name, output_path):
        iframe_files = []
        for idx, frame in enumerate(frames):
            iframe = self.copy_iframe_data(frame, filename, output_path + output_file_name + '-' + str(idx) + '.h264')
            if iframe is not None:
                iframe_files.append(iframe)

        return iframe_files

    @staticmethod
    def copy_frame_data(frame, infile_path, outfile_path):
        return MP4.copy_data(frame.pos, frame.size, infile_path, outfile_path)

 
    @staticmethod
    def copy_data(pos, size, infile_path, outfile_path):
        buffer_size = 4096  # 4 KiB

        with open(infile_path, "rb") as in_file:
            in_file.seek(pos)
            with open(outfile_path, "ab") as out_file:
                toread_len = size
                while toread_len > 0:
                    if toread_len < buffer_size:
                        chunk = in_file.read(toread_len)
                    else:
                        chunk = in_file.read(buffer_size)

                    if chunk == b"":
                        break  # end of file

                    out_file.write(chunk)
                    toread_len -= len(chunk)

    @staticmethod
    def write_frame(filename, frames, output_file_paths):
        frame_files = []
        for idx, frame in enumerate(frames):
            copied_frame = output_file_paths[idx]
            MP4.copy_frame_data(frame, filename, copied_frame )
            frame_files.append(copied_frame)

        return frame_files
    
    @staticmethod
    def write_vcl_nal_frame(filename, frames, output_file_paths):
        frame_files = []
        for idx, frame in enumerate(frames):
            copied_frame = MP4.copy_vcl_nal(frame, filename, output_file_paths[idx])
            if copied_frame is not None:
                frame_files.append(copied_frame)

        return frame_files
    
    @staticmethod
    def copy_vcl_nal( frame, infile_path, outfile_path):
        current_nal_pos  = frame.pos
        frame_size = frame.size
        is_i_frame = False
        i_frames = []
        with open(infile_path, "rb") as in_file:
            while current_nal_pos < frame.pos + frame_size :
                in_file.seek(current_nal_pos)
                is_i_frame = False
                nal_length_byte_length = 0
                nal_length_bytes = in_file.read(4)
                if nal_length_bytes is None or len(nal_length_bytes) < 4:
                    # nal not complete
                    return None

                nal_length_byte_length = 4
                nal_length = struct.unpack('>I', nal_length_bytes)[0]
                if nal_length == 1:
                    nal_length_bytes = in_file.read(8)
                    nal_length = int(struct.unpack('>Q', nal_length_bytes)[0])
                    print('big atom', nal_length, nal_length_bytes)
                    nal_length_byte_length += 8 

                nal_unit_type_bytes = in_file.read(1)
                if nal_unit_type_bytes is None:
                    return None  # nal not complete
                nal_unit_type = nal_unit_type_bytes[0] & 0x1f

                first_byte_in_slice = in_file.read(1)
                if first_byte_in_slice == None:
                    return None

                if nal_unit_type == 5:  # 5: idr frame is i frame
                    #print("idr frame")
                    is_i_frame = True
                    #break
                elif nal_unit_type in [1, 2, 3, 4]:
                    slice_type_byte = first_byte_in_slice[0] & 0b1111111
                    slice_type = MP4.unary_decode(format(slice_type_byte, '07b'))
                    print("nal_unit_type", nal_unit_type, 'slice type', slice_type)
                    if slice_type in [2, 7]:
                        #print("i frame")
                        is_i_frame = True
                    #else:
                        #print("other slide")
                #else: 
                    #print(f"other nal {nal_unit_type}")

                if is_i_frame:
                     i_frames.append((current_nal_pos,  nal_length + nal_length_byte_length))
                    

                current_nal_pos += nal_length + nal_length_byte_length

        if len(i_frames)>0:
            for (pos, size) in i_frames:
                MP4.copy_data(pos, size, infile_path, outfile_path)


        return outfile_path

    def read_iframe_data(self, infile_path):
 
        current_nal_pos  = 0
        file_stats = os.stat(infile_path)
        frame_size = file_stats.st_size
        with open(infile_path, "rb") as in_file:
            while current_nal_pos < frame_size :
                in_file.seek(current_nal_pos)
                nal_length_byte_length = 0
                nal_length_bytes = in_file.read(4)
                if nal_length_bytes is None or len(nal_length_bytes) < 4:
                    # nal not complete
                    return None

                nal_length_byte_length = 4
                nal_length = struct.unpack('>I', nal_length_bytes)[0]
                if nal_length == 1:
                    nal_length_bytes = in_file.read(8)
                    nal_length = int(struct.unpack('>Q', nal_length_bytes)[0])
                    print('big atom', nal_length, nal_length_bytes)
                    nal_length_byte_length += 8 

                nal_unit_type_bytes = in_file.read(1)
                if nal_unit_type_bytes is None:
                    return None  # nal not complete
                nal_unit_type = nal_unit_type_bytes[0] & 0x1f

                first_byte_in_slice = in_file.read(1)
                if first_byte_in_slice == None:
                    return None

                if nal_unit_type == 5:  # 5: idr frame is i frame
                    print("idr frame")
                    #break
                elif nal_unit_type in [1, 2, 3, 4]:
                    slice_type_byte = first_byte_in_slice[0] & 0b1111111
                    slice_type = MP4.unary_decode(format(slice_type_byte, '07b'))
                    print("nal_unit_type", nal_unit_type, 'slice type', slice_type)
                    if slice_type in [2, 7]:
                        print("i frame")
                    else:
                        print("other slide")
                else: 
                    print(f"other nal {nal_unit_type}")
                
                current_nal_pos += nal_length + nal_length_byte_length


        return None
        
    @staticmethod
    def convert_h264_data(infile_path, outfile_path):
 
        file_stats = os.stat(infile_path)

        file_size = file_stats.st_size
   
        is_i_frame = False
        i_frames = []
        framestart = []
        last_three_bytes = b''
        with open(infile_path, "rb") as in_file:
            byte_read = in_file.read(1) 
            while byte_read:
                is_i_frame = False
                last_three_bytes = (last_three_bytes + byte_read)[-3:]
                if last_three_bytes == b'\x00\x00\x01':
                    startpos = in_file.tell() - 3
                    framestart.append(startpos)   
                    
                    nal_unit_type_bytes = in_file.read(1)
            
                    if nal_unit_type_bytes is None:
                        return None  # nal not complete
                    nal_unit_type = nal_unit_type_bytes[0] & 0x1f

                    first_byte_in_slice = in_file.read(1)
        
                    if first_byte_in_slice == None:
                        return None

                    if nal_unit_type == 5:  # 5: idr frame is i frame
                        #print("idr frame")
                        is_i_frame = True
                        #break
                    elif nal_unit_type in [1, 2, 3, 4]:
                        slice_type_byte = first_byte_in_slice[0] & 0b1111111
                        slice_type = MP4.unary_decode(format(slice_type_byte, '07b'))
                        print("nal_unit_type", nal_unit_type, 'slice type', slice_type)
                        if slice_type in [2, 7]:
                            #print("i frame")
                            is_i_frame = True
                        #else:
                            #print("other slide")
                    #else: 
                        #print(f"other nal {nal_unit_type}")

                    if is_i_frame:
                        i_frames.append(startpos)
                
                byte_read = in_file.read(1) 
                    

                

        if len(i_frames)>0:
            for startpos in i_frames:
                frameindex = framestart.index(startpos)
                if frameindex >= len(framestart) - 1:
                    end = file_size
                else:
                    end = framestart[framestart.index(startpos) + 1]
                
                MP4.copy_data(startpos, end - startpos, infile_path, outfile_path)
        else:
            print(infile_path," no iframe")


        return outfile_path


if __name__ == "__main__":
    #mp4 = MP4('C:/dev/video-byteformer/mp4_parser/data/aug_video1.mp4')

    #mp4.parse()
    #mp4.traverse()

    #frames = mp4.get_samples()
    #print('frames count:', len(frames))

    #mp4.write_i_frame('C:/dev/video-byteformer/mp4_parser/data/aug_video1.mp4',\
    #                  frames,\
    #                   'uid2',\
    #                  'C:/dev/video-byteformer/mp4_parser/data/')
    

    #mp4 = MP4('/data/h264_v20231123/h264_old/55ac670d-9a64-411c-8254-05238c62836f-1271230.h264') 
    #dirpath = '/data/h264_v20231123/h264_old/'


    #dirpath = os.path.expanduser(dirpath)
    #json_list = []
    #id_list = []
    #for filename in  os.listdir(dirpath):
    #    fpath = os.path.join(dirpath, filename)
    #    # print(path.encode('utf-8'))
    #    if fpath.endswith('.h264'):
    #        try:
    #            #print(fpath)
    #            mp4.read_iframe_data(fpath)
    #        except OSError:
    #            pass


    MP4.convert_h264_data('/data/h264_v20231126/temp/643702f394763e3cd3a4e226_4.h264', '/data/h264_v20231126/h264/643702f394763e3cd3a4e226_4.h264')
   
    



