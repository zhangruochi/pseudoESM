#!/usr/bin/perl

use strict;
use warnings;


my $ifn = shift || die "USAGE: maskFasta.pl seq95-aa26-part089.fasta MASKED_OUT.fasta";
my $ofn = shift || die "USAGE: maskFasta.pl seq95-aa26-part089.fasta MASKED_OUT.fasta";
my $masklen = 20;


sub mask {
    my ($seq) = @_;
    my $seqlen = length($seq);

    if ($seqlen > $masklen * 3) {
        my $idx = int(rand() * ($seqlen - $masklen * 3)) + $masklen;
        substr($seq, $idx, $masklen) = "<mask>" x $masklen;
    }

    return $seq;
}


my ($header, $buffer) = (undef, "");
open(INPUT, $ifn);
open(OUTPUT, ">$ofn");
while (<INPUT>) {
    if (/^>/) {
        print(OUTPUT $header, mask($buffer), "\n") if $header;
        ($header, $buffer) = ($_, "");
    } else {
        chomp($buffer .= $_);
    }
}
print(OUTPUT $header, mask($buffer), "\n") if $header;
close(OUTPUT);
close(INPUT);

