add_library( getm_memory
    grid.F90
    logging.F90
    memory_manager.F90
)
set_property( TARGET getm_memory PROPERTY FOLDER getm )
