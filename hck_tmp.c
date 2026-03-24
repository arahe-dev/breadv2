#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#define fseek64(f,o,w) _fseeki64((f),(__int64)(o),(w))
#endif
static void xread(FILE*f,void*b,size_t n){if(fread(b,1,n,f)!=n){exit(1);}}
static uint32_t r_u32(FILE*f){uint32_t v;xread(f,&v,4);return v;}
static uint64_t r_u64(FILE*f){uint64_t v;xread(f,&v,8);return v;}
static char* rstr(FILE*f){uint64_t l=r_u64(f);char*s=malloc(l+1);xread(f,s,l);s[l]=0;return s;}
static void skip_str(FILE*f){uint64_t l=r_u64(f);fseek64(f,l,SEEK_CUR);}
static void skip_val(FILE*f,uint32_t t);
static void skip_val(FILE*f,uint32_t t){
    uint64_t cnt; uint32_t et; size_t esz;
    switch(t){case 0:case 1:case 7:fseek64(f,1,SEEK_CUR);return;
    case 2:case 3:fseek64(f,2,SEEK_CUR);return;
    case 4:case 5:case 6:fseek64(f,4,SEEK_CUR);return;
    case 10:case 11:case 12:fseek64(f,8,SEEK_CUR);return;
    case 8:skip_str(f);return;
    case 9:et=r_u32(f);cnt=r_u64(f);
        switch(et){case 0:case 1:case 7:esz=1;break;case 2:case 3:esz=2;break;
        case 4:case 5:case 6:esz=4;break;case 10:case 11:case 12:esz=8;break;default:esz=0;}
        if(esz>0)fseek64(f,cnt*esz,SEEK_CUR);
        else{for(uint64_t i=0;i<cnt;i++)skip_val(f,et);}return;}
}
int main(){
    FILE*f=fopen("C:\\Users\\arahe\\.ollama\\models\\blobs\\sha256-900dde62fb7ebe8a5a25e35d5b7633f403f226a310965fed51d50f5238ba145a","rb");
    r_u32(f);r_u32(f);uint64_t nt=r_u64(f);uint64_t nk=r_u64(f);
    for(uint64_t i=0;i<nk;i++){
        char*k=rstr(f);uint32_t vt=r_u32(f);
        if(strcmp(k,"qwen35moe.attention.head_count_kv")==0&&vt==9){
            uint32_t et=r_u32(f);uint64_t cnt=r_u64(f);
            printf("head_count_kv: array[%llu] type=%u  values: ",(unsigned long long)cnt,(unsigned)et);
            for(uint64_t j=0;j<cnt;j++){uint32_t v=r_u32(f);printf("%u ",v);}
            printf("\n");
        } else {skip_val(f,vt);}
        free(k);
    }
    fclose(f);return 0;
}
