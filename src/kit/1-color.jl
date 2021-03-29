COLORS = Dict(
"black"         => 0,
"red"           => 1,
"green"         => 2,
"yellow"        => 3,
"blue"          => 4,
"magenta"       => 5,
"cyan"          => 6,
"white"         => 7,
"default"       => 9
)


MODES = Dict(
"default"       => 0,
"bold"          => 1,
"underline"     => 4,
"blink"         => 5,
"swap"          => 7,
"hide"          => 8
)


function color(color::String, str::String; background::String="default", mode::String="default")
    FColor = "\e[$(30 + COLORS[color]);"
    BColor =    "$(40 + COLORS[background])m"
    Action = "\e[$(MODES[mode]);m"
    return FColor * BColor * str * Action
end


black(str)   = color("black",   string(str))
red(str)     = color("red",     string(str))
green(str)   = color("green",   string(str))
yellow(str)  = color("yellow",  string(str))
blue(str)    = color("blue",    string(str))
magenta(str) = color("magenta", string(str))
cyan(str)    = color("cyan",    string(str))
white(str)   = color("white",   string(str))
