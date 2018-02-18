import Foundation
import Cocoa

func main() {
    let infoRef = CGWindowListCopyWindowInfo(CGWindowListOption.optionOnScreenOnly, CGWindowID(0))
    let info = infoRef as NSArray? as! [[String: AnyObject]]
    
    for windowInfo in info {
        let windowBounds = windowInfo["kCGWindowBounds"] as! CFDictionary
        let windowRect = CGRect.init(dictionaryRepresentation: windowBounds)!
        
        let windowLayer = windowInfo["kCGWindowLayer"] as! Int
        let windowOwnerName = windowInfo["kCGWindowOwnerName"] as! String?
        
        if windowLayer != 0 { // throw out windows we don't care about
            continue
        }

        if let name = windowInfo["kCGWindowName"] as! String? {
            print(name, terminator: "\t")
            print(windowRect)
        }

        // if windowRect.contains(point) {
        //     // info is ordered front to back so just return first thing we get
        //     windowUnderPoint = (
        //         name: windowInfo["kCGWindowName"] as! String?,
        //         ownerName: windowOwnerName
        //     )
        //     break
        // }
    }
    
}

main()
