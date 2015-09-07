#!/usr/bin/env perl

use utf8;
use open ':encoding(utf8)';
use Encode;

binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");
binmode(STDERR, ":utf8");

$separator="&";

$file=$ARGV[0];
$max=$ARGV[1];
$threshold=$ARGV[2];
$key=$ARGV[3];

open(f,$file);
while(<f>){
  chop;
  @r=split(/\s+/); 
  if (/^KW/){
    $key=$_;
  }
  elsif(/\#/){
    $line=$_;
  }
  elsif(/^[0-9]/ && /\./ && $#r==0){
    $score=$_;
    $score=exp((-1)*$score);
     #print "SCORE=$score\n";
    #print "LINE=$line\n";
    $line=~s/\<epsilon\>//g;
    $line=~s/^\s+//;
    $line=~s/\s+$//;
    $line=~s/\s+/ /;
    #print $_,"\n";
    @a=split(/\s+/,$line); 
    $name=$key;
    $file=$a[0];
    
    $word=$key.$separator;
    for ($i=1; $i<=$#a; $i++){
      if ($a[$i]=~/$separator/){
         @f=split($separator,$a[$i]); 
         $word.=$f[0]."|";
         $a[$i]=$f[1];
      }
    }
    $word=~s/\|$//;
    undef @b; undef @c;
    @b=split(/\-/,$a[1]);
    $start=$b[0];
    @c=split(/\-/,$a[$#a]);
    $end=$c[1];

    if ($#a >0){
      if ($#a  <= $max && $score > $threshold ){
	for ($i=2;$i<=$#a-1; $i++){
	  @b=split(/\-/,$a[$i]);
	  if ($b[0]< $start){ $start=$b[0]};
	  if ($b[1]> $end){$end=$b[1]};
	}
	if ($end<=$start){
	  $cc=$start;
	  $start=$end;
	  $end=$cc;
	}
      print  $file," ",$start," ",$end," ",$word," ",$score,"\n";
    }
  }
  }
  else{
    #$word=$_;
    #$word=~s/\s+/\|/g; $word=~s/\|$//g; 
  }
}

close(f);
