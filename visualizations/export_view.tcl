proc save_view_settings {{outfile "view_settings.tcl"}} {
    if {[molinfo num] == 0} {
        puts "❌ No molecule loaded. Cannot save view settings."
        return
    }
    puts "💾 Saving view settings to '$outfile'..."

    set molid [molinfo top]
    set out [open $outfile w]
    puts $out "# Saved VMD view settings"

    set rotate_matrix [molinfo $molid get rotate_matrix]
    set center        [molinfo $molid get center]
    set global_matrix [molinfo $molid get global_matrix]
    set scale_matrix  [molinfo $molid get scale_matrix]

    puts $out "molinfo top set rotate_matrix [list $rotate_matrix]"
    puts $out "molinfo top set center [list $center]"
    puts $out "molinfo top set global_matrix [list $global_matrix]"
    puts $out "molinfo top set scale_matrix [list $scale_matrix]"

    close $out
    puts "✔ View settings saved to '$outfile'"
}
