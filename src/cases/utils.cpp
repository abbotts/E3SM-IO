/*********************************************************************
 *
 * Copyright (C) 2021, Northwestern University
 * See COPYRIGHT notice in top-level directory.
 *
 * This program is part of the E3SM I/O benchmark.
 *
 *********************************************************************/
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <mpi.h>

#include <e3sm_io.h>
#include <e3sm_io_case.hpp>
#include <e3sm_io_driver.hpp>

#ifdef HAVE_HIP
#include <hip/hip_runtime_api.h>
#define HIP_RC(e) if((e) != hipSuccess) assert((e != hipSuccess) && "hipError Encountered)")
#endif

/*----< check_malloc() >-----------------------------------------------------*/
int e3sm_io_case::check_malloc(e3sm_io_config *cfg,
                               e3sm_io_driver *driver)
{
    int err=0, global_rank;
    MPI_Offset m_alloc, s_alloc, x_alloc;

    if (!cfg->verbose || cfg->api != pnetcdf) return 0;

    MPI_Comm_rank(cfg->io_comm, &global_rank);

    /* check if there is any PnetCDF internal malloc residue */
    err = driver->inq_malloc_size(&m_alloc);
    if (err == NC_NOERR) {
        MPI_Reduce(&m_alloc, &s_alloc, 1, MPI_OFFSET, MPI_SUM, 0, cfg->io_comm);
        if (global_rank == 0 && s_alloc > 0) {
            printf("-------------------------------------------------------\n");
            printf("Residue heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   s_alloc);
        }
    }

    /* find the high water mark among all processes */
    driver->inq_malloc_max_size(&m_alloc);
    MPI_Reduce(&m_alloc, &x_alloc, 1, MPI_OFFSET, MPI_MAX, 0, cfg->io_comm);
    if (global_rank == 0)
        printf("High water mark of heap memory allocated by PnetCDF internally is %.2f MiB\n",
               (float)x_alloc / 1048576);

    return err;
}

/*----< wr_buf_init() >------------------------------------------------------*/
void e3sm_io_case::wr_buf_init(int gap)
{
    wr_buf.gap = gap;

    wr_buf.fix_txt_buflen = 0;
    wr_buf.fix_int_buflen = 0;
    wr_buf.fix_flt_buflen = 0;
    wr_buf.fix_dbl_buflen = 0;
    wr_buf.fix_lld_buflen = 0;
    wr_buf.rec_txt_buflen = 0;
    wr_buf.rec_int_buflen = 0;
    wr_buf.rec_flt_buflen = 0;
    wr_buf.rec_dbl_buflen = 0;
    wr_buf.rec_lld_buflen = 0;

    wr_buf.fix_txt_buf = NULL;
    wr_buf.fix_int_buf = NULL;
    wr_buf.fix_flt_buf = NULL;
    wr_buf.fix_dbl_buf = NULL;
    wr_buf.fix_lld_buf = NULL;
    wr_buf.rec_txt_buf = NULL;
    wr_buf.rec_int_buf = NULL;
    wr_buf.rec_flt_buf = NULL;
    wr_buf.rec_dbl_buf = NULL;
    wr_buf.rec_lld_buf = NULL;
}

static void *dispatch_allocator(int mem_type, size_t size)
{
    int err;
    void *rval;
    switch (mem_type) {
    case MT_MALLOC:
	rval = malloc(size);
	break;
#ifdef HAVE_HIP
    case MT_HIPMALLOC:
	HIP_RC(hipMalloc(&rval, size));
	break;
    case MT_HIPHOST:
	HIP_RC(hipHostMalloc(&rval, size, hipHostMallocDefault));
	break;
    case MT_HIPMANAGED:
	HIP_RC(hipMallocManaged(&rval, size, hipMemAttachGlobal));
	break;
#endif
    default:	
      assert( 0 && "Unrecognized memory type requested.");
    }
    assert(rval != NULL);
    return rval;
}

static void dispatch_free(int memtype, void *ptr)
{
    int err;
    switch (memtype) {
    case MT_MALLOC:
	free(ptr);
	break;
#ifdef HAVE_HIP
    case MT_HIPMALLOC:
    case MT_HIPHOST:
    case MT_HIPMANAGED:
	HIP_RC(hipFree(ptr));
	break;
#endif
    default:	
      assert(0 && "Unrecognized memory type requested.");
    }    
}

/*----< wr_buf_malloc() >----------------------------------------------------*/
int e3sm_io_case::wr_buf_malloc(e3sm_io_config &cfg, int ffreq)
{
    int rank;
    size_t j;

    MPI_Comm_rank(cfg.io_comm, &rank);

    if (cfg.api == adios) {
        wr_buf.fix_txt_buflen += 64;
        wr_buf.fix_int_buflen += 64;
        wr_buf.fix_flt_buflen += 64;
        wr_buf.fix_dbl_buflen += 64;
        wr_buf.fix_lld_buflen += 64;
        wr_buf.rec_txt_buflen += 64;
        wr_buf.rec_int_buflen += 64;
        wr_buf.rec_flt_buflen += 64;
        wr_buf.rec_dbl_buflen += 64;
        wr_buf.rec_lld_buflen += 64;
    }

    if (cfg.api != adios && !(cfg.strategy == blob && cfg.api == hdf5)) {
        /* Note HDF5 and ADIOS blob I/O copy write data into their internal
         * buffers and only flush them out at file close. Thus, write buffers
         * can be reused for these two I/O methods. For others, such as PnetCDF
         * and HDF5 log-based VOL, write buffers should not be touched as they
         * will later be used during the flushing is called.
         */
        wr_buf.rec_txt_buflen *= ffreq;
        wr_buf.rec_int_buflen *= ffreq;
        wr_buf.rec_flt_buflen *= ffreq;
        wr_buf.rec_dbl_buflen *= ffreq;
        wr_buf.rec_lld_buflen *= ffreq;
    }

    /* allocate and initialize write buffers */
    if (cfg.non_contig_buf) {
        wr_buf.fix_txt_buf = (char*)   dispatch_allocator(cfg.mem_type, wr_buf.fix_txt_buflen * sizeof(char));
        wr_buf.fix_int_buf = (int*)    dispatch_allocator(cfg.mem_type, wr_buf.fix_int_buflen * sizeof(int));
        wr_buf.fix_flt_buf = (float*)  dispatch_allocator(cfg.mem_type, wr_buf.fix_flt_buflen * sizeof(float));
        wr_buf.fix_dbl_buf = (double*) dispatch_allocator(cfg.mem_type, wr_buf.fix_dbl_buflen * sizeof(double));
        wr_buf.fix_lld_buf = (long long*) dispatch_allocator(cfg.mem_type, wr_buf.fix_lld_buflen * sizeof(long long));
        wr_buf.rec_txt_buf = (char*)   dispatch_allocator(cfg.mem_type, wr_buf.rec_txt_buflen * sizeof(char));
        wr_buf.rec_int_buf = (int*)    dispatch_allocator(cfg.mem_type, wr_buf.rec_int_buflen * sizeof(int));
        wr_buf.rec_flt_buf = (float*)  dispatch_allocator(cfg.mem_type, wr_buf.rec_flt_buflen * sizeof(float));
        wr_buf.rec_dbl_buf = (double*) dispatch_allocator(cfg.mem_type, wr_buf.rec_dbl_buflen * sizeof(double));
        wr_buf.rec_lld_buf = (long long*) dispatch_allocator(cfg.mem_type, wr_buf.rec_lld_buflen * sizeof(long long));
    }
    else {
        size_t sum = wr_buf.fix_txt_buflen
                   + wr_buf.fix_int_buflen * sizeof(int)
                   + wr_buf.fix_dbl_buflen * sizeof(double)
                   + wr_buf.fix_flt_buflen * sizeof(float)
                   + wr_buf.fix_lld_buflen * sizeof(long long)
                   + wr_buf.rec_txt_buflen
                   + wr_buf.rec_int_buflen * sizeof(int)
                   + wr_buf.rec_dbl_buflen * sizeof(double)
                   + wr_buf.rec_flt_buflen * sizeof(float)
                   + wr_buf.rec_lld_buflen * sizeof(long long);

        wr_buf.fix_txt_buf = (char*) dispatch_allocator(cfg.mem_type, sum);
        wr_buf.fix_int_buf = (int*)      (wr_buf.fix_txt_buf + wr_buf.fix_txt_buflen);
        wr_buf.fix_dbl_buf = (double*)   (wr_buf.fix_int_buf + wr_buf.fix_int_buflen);
        wr_buf.fix_flt_buf = (float*)    (wr_buf.fix_dbl_buf + wr_buf.fix_dbl_buflen);
        wr_buf.fix_lld_buf = (long long*)(wr_buf.fix_flt_buf + wr_buf.fix_flt_buflen);

        wr_buf.rec_txt_buf = (char*)     (wr_buf.fix_lld_buf + wr_buf.fix_lld_buflen);
        wr_buf.rec_int_buf = (int*)      (wr_buf.rec_txt_buf + wr_buf.rec_txt_buflen);
        wr_buf.rec_dbl_buf = (double*)   (wr_buf.rec_int_buf + wr_buf.rec_int_buflen);
        wr_buf.rec_flt_buf = (float*)    (wr_buf.rec_dbl_buf + wr_buf.rec_dbl_buflen);
        wr_buf.rec_lld_buf = (long long*)(wr_buf.rec_flt_buf + wr_buf.rec_flt_buflen);
    }

    for (j=0; j<wr_buf.fix_txt_buflen; j++) wr_buf.fix_txt_buf[j] = 'a' + rank;
    for (j=0; j<wr_buf.fix_int_buflen; j++) wr_buf.fix_int_buf[j] = rank;
    for (j=0; j<wr_buf.fix_dbl_buflen; j++) wr_buf.fix_dbl_buf[j] = rank;
    for (j=0; j<wr_buf.fix_flt_buflen; j++) wr_buf.fix_flt_buf[j] = rank;
    for (j=0; j<wr_buf.fix_lld_buflen; j++) wr_buf.fix_lld_buf[j] = rank;
    for (j=0; j<wr_buf.rec_txt_buflen; j++) wr_buf.rec_txt_buf[j] = 'a' + rank;
    for (j=0; j<wr_buf.rec_int_buflen; j++) wr_buf.rec_int_buf[j] = rank;
    for (j=0; j<wr_buf.rec_dbl_buflen; j++) wr_buf.rec_dbl_buf[j] = rank;
    for (j=0; j<wr_buf.rec_flt_buflen; j++) wr_buf.rec_flt_buf[j] = rank;
    for (j=0; j<wr_buf.rec_lld_buflen; j++) wr_buf.rec_lld_buf[j] = rank;

    return 0;
}

/*----< wr_buf_free() >------------------------------------------------------*/
void e3sm_io_case::wr_buf_free(e3sm_io_config &cfg)
{
    if (cfg.non_contig_buf) {
        if (wr_buf.fix_txt_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.fix_txt_buf);
        if (wr_buf.fix_int_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.fix_int_buf);
        if (wr_buf.fix_flt_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.fix_flt_buf);
        if (wr_buf.fix_dbl_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.fix_dbl_buf);
        if (wr_buf.fix_lld_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.fix_lld_buf);
        if (wr_buf.rec_txt_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.rec_txt_buf);
        if (wr_buf.rec_int_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.rec_int_buf);
        if (wr_buf.rec_flt_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.rec_flt_buf);
        if (wr_buf.rec_dbl_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.rec_dbl_buf);
        if (wr_buf.rec_lld_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.rec_lld_buf);
    }
    else {
        if (wr_buf.fix_txt_buf != NULL) dispatch_free(cfg.mem_type, wr_buf.fix_txt_buf);
    }

    wr_buf.fix_txt_buf = NULL;
    wr_buf.fix_int_buf = NULL;
    wr_buf.fix_flt_buf = NULL;
    wr_buf.fix_dbl_buf = NULL;
    wr_buf.fix_lld_buf = NULL;
    wr_buf.rec_txt_buf = NULL;
    wr_buf.rec_int_buf = NULL;
    wr_buf.rec_flt_buf = NULL;
    wr_buf.rec_dbl_buf = NULL;
    wr_buf.rec_lld_buf = NULL;

    wr_buf.fix_txt_buflen = 0;
    wr_buf.fix_int_buflen = 0;
    wr_buf.fix_flt_buflen = 0;
    wr_buf.fix_dbl_buflen = 0;
    wr_buf.fix_lld_buflen = 0;
    wr_buf.rec_txt_buflen = 0;
    wr_buf.rec_int_buflen = 0;
    wr_buf.rec_flt_buflen = 0;
    wr_buf.rec_dbl_buflen = 0;
    wr_buf.rec_lld_buflen = 0;
}

