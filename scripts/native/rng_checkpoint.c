#include <stdio.h>
#include <assert.h>
int randk(void);
void randk_reset_for_test(void);
void randk_seed_for_test(unsigned int);
int randk_write_for_test(FILE *);
int randk_read_for_test(FILE *);
int main(void) {
    int a[64], b[64], differs=0;
    randk_reset_for_test(); for(int i=0;i<64;i++) a[i]=randk();
    randk_seed_for_test(0); for(int i=0;i<64;i++) assert(a[i]==randk());
    randk_seed_for_test(101); for(int i=0;i<64;i++) a[i]=randk();
    randk_seed_for_test(101); for(int i=0;i<64;i++) assert(a[i]==randk());
    randk_seed_for_test(102); for(int i=0;i<64;i++) {b[i]=randk(); differs |= a[i]!=b[i];} assert(differs);
    FILE *f=tmpfile(); assert(f); assert(randk_write_for_test(f));
    for(int i=0;i<64;i++) a[i]=randk();
    rewind(f); assert(randk_read_for_test(f)); for(int i=0;i<64;i++) assert(a[i]==randk());
    fclose(f); f=tmpfile(); assert(f); assert(!randk_read_for_test(f)); fclose(f);
    puts("PASS: equal seeds, distinct seeds, legacy zero, exact RNG restore, truncated save rejection");
}
