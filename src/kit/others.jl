function blocksize(n::Int, u::String)
    if u == "B"  return n / 1 end
    if u == "KB" return n / 1024 end
    if u == "MB" return n / 1048576 end
    if u == "GB" return n / 1073741824 end
    if u == "TB" return n / 1099511627776 end
end
