/* config_reader.c — read architecture metadata from GGUF file
 * Prints key model hyperparameters needed for inference.
 * Compile: cl.exe /O2 /nologo config_reader.c /Fe:config_reader.exe
 * Run:     config_reader.exe <gguf_path>
 */
#ifdef _WIN32
#  define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#  define fseek64(f,o,w) _fseeki64((f),(__int64)(o),(w))
#else
#  define fseek64(f,o,w) fseeko((f),(off_t)(o),(w))
#endif

static void xread(FILE *fp, void *buf, size_t n) {
    if (fread(buf,1,n,fp)!=n){fprintf(stderr,"read err\n");exit(1);}
}
static uint8_t  r_u8 (FILE*f){uint8_t  v;xread(f,&v,1);return v;}
static int8_t   r_i8 (FILE*f){int8_t   v;xread(f,&v,1);return v;}
static uint16_t r_u16(FILE*f){uint16_t v;xread(f,&v,2);return v;}
static int16_t  r_i16(FILE*f){int16_t  v;xread(f,&v,2);return v;}
static uint32_t r_u32(FILE*f){uint32_t v;xread(f,&v,4);return v;}
static int32_t  r_i32(FILE*f){int32_t  v;xread(f,&v,4);return v;}
static uint64_t r_u64(FILE*f){uint64_t v;xread(f,&v,8);return v;}
static int64_t  r_i64(FILE*f){int64_t  v;xread(f,&v,8);return v;}
static float    r_f32(FILE*f){float    v;xread(f,&v,4);return v;}
static double   r_f64(FILE*f){double   v;xread(f,&v,8);return v;}

static char *read_str(FILE *fp) {
    uint64_t len = r_u64(fp);
    char *s = (char*)malloc(len+1);
    xread(fp,s,(size_t)len); s[len]='\0'; return s;
}
static void skip_str(FILE *fp) {
    uint64_t len = r_u64(fp);
    fseek64(fp,(int64_t)len,SEEK_CUR);
}

static void skip_val(FILE *fp, uint32_t vt) {
    switch(vt){
        case 0:case 1:case 7: r_u8(fp);  return;
        case 2:case 3:        r_u16(fp); return;
        case 4:case 5:case 6: r_u32(fp); return;
        case 10:case 11:case 12: r_u64(fp); return;
        case 8: skip_str(fp); return;
        case 9: {
            uint32_t et=r_u32(fp); uint64_t cnt=r_u64(fp);
            size_t esz=0;
            switch(et){
                case 0:case 1:case 7: esz=1; break;
                case 2:case 3:        esz=2; break;
                case 4:case 5:case 6: esz=4; break;
                case 10:case 11:case 12: esz=8; break;
            }
            if(esz>0) fseek64(fp,(int64_t)(cnt*esz),SEEK_CUR);
            else for(uint64_t i=0;i<cnt;i++) skip_val(fp,et);
            return;
        }
    }
}

int main(int argc, char **argv) {
    const char *path = argc>1 ? argv[1] :
        "C:\\Users\\arahe\\.ollama\\models\\blobs\\"
        "sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a";

    FILE *fp = fopen(path,"rb");
    if(!fp){fprintf(stderr,"cannot open %s\n",path);return 1;}

    uint32_t magic=r_u32(fp);
    if(magic!=0x46554747u){fprintf(stderr,"not GGUF\n");return 1;}
    uint32_t version=r_u32(fp);
    uint64_t n_tensors=r_u64(fp);
    uint64_t n_kv=r_u64(fp);
    printf("GGUF v%u  tensors=%llu  kv_pairs=%llu\n",
           (unsigned)version,(unsigned long long)n_tensors,(unsigned long long)n_kv);

    for(uint64_t ki=0; ki<n_kv; ki++) {
        char *key = read_str(fp);
        uint32_t vtype = r_u32(fp);
        int printed = 0;

        /* Print all keys that look architecture-related */
        if(strstr(key,"block_count")||strstr(key,"embedding_length")||
           strstr(key,"attention.head")||strstr(key,"rope.freq")||
           strstr(key,"rms_epsilon")||strstr(key,"expert")||
           strstr(key,"context_length")||strstr(key,"key_length")||
           strstr(key,"value_length")||strstr(key,"feed_forward")||
           strcmp(key,"general.architecture")==0) {

            printf("  %-60s = ", key);
            switch(vtype) {
                case 0: printf("uint8(%u)",  (unsigned)r_u8(fp));  printed=1; break;
                case 1: printf("int8(%d)",   (int)r_i8(fp));       printed=1; break;
                case 2: printf("uint16(%u)", (unsigned)r_u16(fp)); printed=1; break;
                case 3: printf("int16(%d)",  (int)r_i16(fp));      printed=1; break;
                case 4: printf("uint32(%u)", (unsigned)r_u32(fp)); printed=1; break;
                case 5: printf("int32(%d)",  r_i32(fp));           printed=1; break;
                case 6: printf("float32(%g)",r_f32(fp));           printed=1; break;
                case 7: printf("bool(%u)",   (unsigned)r_u8(fp));  printed=1; break;
                case 8: { char*s=read_str(fp); printf("string(\"%s\")",s); free(s); printed=1; break; }
                case 10: printf("uint64(%llu)",(unsigned long long)r_u64(fp)); printed=1; break;
                case 11: printf("int64(%lld)", (long long)r_i64(fp));         printed=1; break;
                case 12: printf("float64(%g)", r_f64(fp));                    printed=1; break;
                default: printf("(array/other vtype=%u)", vtype); printed=0;
            }
            if(printed) printf("\n");
        }

        if(!printed) skip_val(fp, vtype);
        free(key);
    }
    fclose(fp);
    return 0;
}
