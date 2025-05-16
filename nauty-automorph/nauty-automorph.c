#include <stdio.h>
#include "nauty.h"
#include "naugroup.h"


FILE *automorph_file;

void writeautom(int *p, int n)
/* Called by allgroup.  Writes the permutation p to file. */
{
    int i;

    for (i = 0; i < n; ++i)
        fprintf(automorph_file, " %2d", p[i]);
    fprintf(automorph_file, "\n");
}

int
main(int argc, char *argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Missing program arguments. Argument 1 is the vertex number, "
                "argument 2 is the edge list file and argument 3 is output file.\n");
        exit(1);
    }

    FILE *graph_file;

    graph_file = fopen(argv[2], "r");
    automorph_file = fopen(argv[3], "w");

    if (graph_file == NULL) {
        fprintf(stderr, "Graph file %s couldn't be opened.\n", argv[2]); exit(1);
    }
    if (automorph_file == NULL) {
        fprintf(stderr, "Output file %s couldn't be opened.\n", argv[3]); exit(1);
    }


    DYNALLSTAT(graph,g,g_sz);
    DYNALLSTAT(int,lab,lab_sz);
    DYNALLSTAT(int,ptn,ptn_sz);
    DYNALLSTAT(int,orbits,orbits_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;

    int n,m;
    sscanf(argv[1], "%d", &n);

    grouprec *group;

 /* The following cause nauty to call two procedures which
        store the group information as nauty runs. */

    options.userautomproc = groupautomproc;
    options.userlevelproc = grouplevelproc;

    m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);

    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    DYNALLOC1(int,lab,lab_sz,n,"malloc");
    DYNALLOC1(int,ptn,ptn_sz,n,"malloc");
    DYNALLOC1(int,orbits,orbits_sz,n,"malloc");

    EMPTYGRAPH(g,m,n);

    // Read edge list and add edges
    const unsigned MAX_LENGTH = 256;
    char buffer[MAX_LENGTH];

    while (fgets(buffer, MAX_LENGTH, graph_file)) {
        if (buffer[0] == '#' || buffer[0] == '\n')
            continue;

        int v[2] = {n, n};
        int v_len = 0;
        char v_str[MAX_LENGTH];
        int v_i=0;

        for (size_t i=0; i<MAX_LENGTH; i++) {
            if (buffer[i] == ' ' || buffer[i] == '\n' || buffer[i]=='\0') {
                if (v_len > 0) {
                    v_str[v_len] = '\0';
                    sscanf(v_str, "%d", &v[v_i]);
                    v_len = 0;
                    v_i++;

                    if (v_i == 2) break;
                }
            }
            else {
                v_str[v_len++] = buffer[i];
            }
        }
        if (v[0] >= n || v[1] >= n) {
            fprintf(stderr, "Invalid edge list file.\n"); exit(1);
        }
        ADDONEEDGE(g, v[0], v[1], m);
    }

    densenauty(g,lab,ptn,orbits,&options,&stats,m,n,NULL);

 /* Get a pointer to the structure in which the group information
    has been stored.  If you use TRUE as an argument, the
    structure will be "cut loose" so that it won't be used
    again the next time nauty() is called.  Otherwise, as
    here, the same structure is used repeatedly. */

    group = groupptr(FALSE);

 /* Expand the group structure to include a full set of coset
    representatives at every level.  This step is necessary
    if allgroup() is to be called. */

    makecosetreps(group);

 /* Call the procedure writeautom() for every element of the group.
    The first call is always for the identity. */

    allgroup(group,writeautom);

    fclose(graph_file);
    fclose(automorph_file);

    exit(0);
}
